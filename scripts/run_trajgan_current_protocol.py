#!/usr/bin/env python
"""TrajGAN baseline, current protocol, per the PI's instruction (2026-11-08).
One (dataset) per invocation. Mirrors run_gi_dp_current_protocol.py's
shape so every downstream analysis script works unchanged.

The generator trains ONCE on the REAL train split (seed=0, fixed) and
produces ONE synthetic counterpart of the real test split -- there is
no synthetic train release (run_trajgan's own design: the generator is
fit once on real historical data; only the test-period release is
synthetic). Per the PI's explicit choice (2026-11-08), given that
asymmetry:

Utility: next-location Acc@1 of the SAME predictor architecture
(FixedEncoder + MLPHead) used everywhere else, trained on the REAL
train windows (no mechanism applied to training, since no synthetic
train exists), evaluated on the SYNTHETIC test windows against the
TRUE (real) next-location labels -- same "what does a consumer of the
release actually get" question as GI-DP, just answered with the real
train split standing in where GI-DP would have had a perturbed train.
3 predictor-init seeds, same early-stopping protocol as everywhere else.

Risk: the SAME strongest-per-user external attacker, trained AND tested
on an internal 50/50 CHRONOLOGICAL split carved out of the synthetic
test release itself (there is no synthetic train to split against
instead) -- the earlier half of the release is attacker-train, the
later half attacker-test, via the same _chronological_val_split utility
used for early-stopping elsewhere in this project.

The generator's own grid-cell tessellation is NOT reused here (it
builds an independent mapping per call, the same bug class fixed in
run_gi_dp's cell-mapping reuse); this script re-tessellates the
synthetic release with the ORIGINAL data's mapping, extended for any
new cells the generator's lat/lon output lands in outside the original
footprint, mirroring GI-DP's fix.
"""
import os

_N_THREADS = int(os.environ.get("PUFFIN_NUM_THREADS", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_N_THREADS))

import argparse
import copy
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(_N_THREADS)

sys.path.insert(0, ".")
from puffin.baselines.trajgan import run_trajgan
from puffin.data.tessellation import assign_grid_cells
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_strongest_per_user, _chronological_val_split
from puffin.models.mopae_torch import FixedEncoder, MLPHead


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def train_predictor(real_train, true_next_cell_tr, embedding_vocab, output_vocab, device, seed,
                     max_epochs=80, patience=8, min_epochs=20, val_frac=0.1, batch_size=512, lr=3e-4):
    torch.manual_seed(seed)
    sl = real_train["cell"].shape[1]
    encoder = FixedEncoder(sl, embedding_vocab).to(device)
    head = MLPHead(output_vocab).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    n = len(real_train["user"])
    rng = np.random.default_rng(seed)
    if "ts" in real_train:
        val_idx_np, fit_idx_np = _chronological_val_split(real_train["ts"], real_train["user"], val_frac, rng)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(val_frac * n))
        val_idx_np, fit_idx_np = perm[:n_val], perm[n_val:]

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr = to_t(real_train["cell"]), to_t(real_train["weekday"]), to_t(real_train["hour"])
    y = to_t(true_next_cell_tr)
    fit_idx, val_idx = to_t(fit_idx_np), to_t(val_idx_np)

    best_val_acc, best_state, bad_epochs, actual_epochs = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        actual_epochs = epoch + 1
        encoder.train(); head.train()
        perm_t = fit_idx[torch.randperm(len(fit_idx), device=device)]
        for start in range(0, len(perm_t), batch_size):
            idx = perm_t[start:start + batch_size]
            opt.zero_grad()
            logits = head(encoder(cell[idx], wd[idx], hr[idx]))
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            opt.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            val_logits = head(encoder(cell[val_idx], wd[val_idx], hr[val_idx]))
            val_acc = (val_logits.argmax(-1) == y[val_idx]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if actual_epochs >= min_epochs and bad_epochs >= patience:
                break
    encoder.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    return encoder, head, actual_epochs


def eval_utility(encoder, head, synthetic_test, true_next_cell_te, device, batch_size=1024):
    encoder.eval(); head.eval()
    n = len(synthetic_test["user"])
    correct = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl_ = slice(start, start + batch_size)
            cell = torch.tensor(synthetic_test["cell"][sl_], dtype=torch.long, device=device)
            wd = torch.tensor(synthetic_test["weekday"][sl_], dtype=torch.long, device=device)
            hr = torch.tensor(synthetic_test["hour"][sl_], dtype=torch.long, device=device)
            y = torch.tensor(true_next_cell_te[sl_], dtype=torch.long, device=device)
            logits = head(encoder(cell, wd, hr))
            correct += (logits.argmax(-1) == y).sum().item()
    return correct / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-privacy-run-dir", default=None,
                     help="reuse raw_top1 from this existing run instead of recomputing")
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--trajgan-epochs", type=int, default=60)
    ap.add_argument("--trajgan-seed", type=int, default=0)
    ap.add_argument("--encoder-seeds", type=str, default="0,1,2")
    ap.add_argument("--attacker-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--profiled-epoch-seconds", type=float, default=None,
                     help="isolated-profile epoch time for this dataset; if the real "
                          "mean epoch time exceeds 10x this, stop and report instead "
                          "of continuing into the (expensive) attacker phases")
    args = ap.parse_args()
    device = torch.device(args.device)
    encoder_seeds = tuple(int(s) for s in args.encoder_seeds.split(","))
    attacker_seeds = tuple(int(s) for s in args.attacker_seeds.split(","))
    tag = f"[{args.dataset} trajgan]"

    _stage_t0 = time.time()
    df = pd.read_parquet(args.parquet)
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    total_users = int(df["user_id"].max())
    df = assign_grid_cells(df, resolution_deg=0.01)
    original_mapping = df.attrs["grid_cell_mapping"]
    original_total_locations = max(original_mapping.values())
    print(f"{tag} total_locations={original_total_locations}, building ORIGINAL windows...", flush=True)

    original_windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                         respect_traj_boundary=False)
    # PI-caught bug (2026-11-09, confirmed via diagnostic): build_sl_windows
    # shares ONE sequentially-advancing rng across train+test, so
    # original_windows["test"]'s random per-user subsample draws happen
    # AFTER consuming rng state from the entire train split's sampling --
    # while the synthetic release below is windowed in ISOLATION (a
    # single-split call, fresh rng). That rng-position mismatch (not
    # chunking, not ordering -- the underlying point sets matched exactly,
    # confirmed by set comparison) is what broke the alignment assertion.
    # Recompute the TRUE test windows the same isolated way so both sides
    # start from equivalent rng state.
    _test_only = df[df["split"] == "test"].copy()
    _test_only["split"] = "train"
    original_windows["test"] = build_sl_windows(
        _test_only, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
        respect_traj_boundary=False)["train"]
    prep_stage_seconds = time.time() - _stage_t0
    print(f"{tag} STAGE windowing/prep: {prep_stage_seconds:.1f}s", flush=True)

    print(f"{tag} training LSTM-TrajGAN (seed={args.trajgan_seed}, {args.trajgan_epochs} epochs)...", flush=True)
    t0 = time.time()
    cache_dir = pathlib.Path(args.out_dir).parent / "_trajgan_groups_cache"
    synthetic_df, trajgan_summary = run_trajgan(
        df, device, max_traj_len=150, min_traj_len=args.sl + 1,
        max_epochs=args.trajgan_epochs, seed=args.trajgan_seed,
        cache_dir=cache_dir, cache_key=args.dataset)
    gan_stage_seconds = time.time() - t0
    mean_epoch_s = trajgan_summary["mean_epoch_time_seconds"]
    print(f"{tag} STAGE generator training: {gan_stage_seconds:.1f}s total, "
          f"{mean_epoch_s:.2f}s/epoch mean over {len(trajgan_summary['epoch_times_seconds'])} epochs, "
          f"n_train_chunks={trajgan_summary['n_train_chunks']}, n_test_chunks={trajgan_summary['n_test_chunks']}, "
          f"final_d_loss={trajgan_summary['final_d_loss']:.4f}, final_g_loss={trajgan_summary['final_g_loss']:.4f}, "
          f"weekday_acc={trajgan_summary['weekday_argmax_accuracy']:.4f}, "
          f"hour_acc={trajgan_summary['hour_argmax_accuracy']:.4f}", flush=True)
    print(f"{tag} STAGE synthetic generation: {trajgan_summary['generation_time_seconds']:.1f}s", flush=True)

    if args.profiled_epoch_seconds is not None and mean_epoch_s > 10 * args.profiled_epoch_seconds:
        raise RuntimeError(
            f"{tag} STOP: mean epoch time {mean_epoch_s:.2f}s is more than 10x the profiled "
            f"rate ({args.profiled_epoch_seconds:.2f}s) -- per the PI's instruction (2026-11-09), "
            f"halting before the attacker phases rather than continuing into what caused the "
            f"original 7.2h/2-epoch slowdown. Report this and investigate before relaunching.")

    # Re-tessellate with the ORIGINAL mapping, extended for any new cells --
    # run_trajgan's own internal assign_grid_cells call builds an independent
    # mapping, which would make cell ids incomparable to the true labels.
    synthetic_df = synthetic_df[["user_id", "traj_id", "ts", "lat", "lon", "weekday", "hour"]].copy()
    synthetic_df["split"] = "train"  # arbitrary single-split label; this IS the whole release
    synthetic_df = assign_grid_cells(synthetic_df, resolution_deg=0.01, existing_mapping=original_mapping)
    embedding_vocab = max(original_total_locations, int(synthetic_df["cell_id"].max()))
    print(f"{tag} embedding vocab after synthesis: {embedding_vocab} "
          f"(output vocab stays at the original {original_total_locations})", flush=True)

    synth_windows_raw = build_sl_windows(synthetic_df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                          respect_traj_boundary=False)
    synthetic_test = synth_windows_raw["train"]  # the single split we labeled "train" above
    assert np.array_equal(synthetic_test["user"], original_windows["test"]["user"]), \
        f"{tag} window alignment broke between real test and synthetic release -- cannot pair true labels"
    assert np.array_equal(synthetic_test["ts"], original_windows["test"]["ts"]), \
        f"{tag} window alignment broke (ts)"
    print(f"{tag} window alignment verified ({len(synthetic_test['user'])} synthetic test windows, "
          f"{len(original_windows['train']['user'])} real train windows)", flush=True)

    # Attacker's own internal split: carved out of the synthetic release itself
    # (no synthetic train exists), earlier half by time = attacker-train.
    attacker_rng = np.random.default_rng(0)
    atk_test_idx, atk_train_idx = _chronological_val_split(
        synthetic_test["ts"], synthetic_test["user"], val_frac=0.5, rng=attacker_rng)
    attacker_windows = {
        "train": {k: v[atk_train_idx] for k, v in synthetic_test.items()},
        "test": {k: v[atk_test_idx] for k, v in synthetic_test.items()},
    }
    print(f"{tag} attacker internal split: {len(atk_train_idx)} train / {len(atk_test_idx)} test "
          f"(carved from the synthetic release)", flush=True)

    feats = compute_mobility_features(df[df["split"] == "train"])
    entropy_by_user = dict(zip(feats.user_id, feats.s_unc))

    if args.no_privacy_run_dir:
        raw_df = pd.read_csv(pathlib.Path(args.no_privacy_run_dir) / "per_user_04d.csv")
        raw_lookup = dict(zip(raw_df.user_id, raw_df.raw_top1))
    else:
        raw_lookup = {}

    utilities, actual_epochs_list, per_encoder_seed_post = [], [], []
    for enc_seed in encoder_seeds:
        t0 = time.time()
        print(f"{tag} encoder seed {enc_seed}: training predictor on REAL train...", flush=True)
        encoder, head, actual_epochs = train_predictor(
            original_windows["train"], original_windows["train"]["next_cell"],
            embedding_vocab, original_total_locations, device, enc_seed,
            max_epochs=args.max_epochs, patience=args.patience, min_epochs=args.min_epochs, val_frac=args.val_frac)
        utility = eval_utility(encoder, head, synthetic_test, original_windows["test"]["next_cell"], device)
        utilities.append(utility)
        actual_epochs_list.append(actual_epochs)
        print(f"{tag} encoder seed {enc_seed}: stopped after {actual_epochs} epochs, utility={utility:.4f}, "
              f"post attacker ({len(attacker_seeds)} seeds) on synthetic release...", flush=True)

        post_df, _ = train_raw_attacker_strongest_per_user(
            attacker_windows, embedding_vocab, total_users, device, seeds=attacker_seeds,
            max_epochs=args.attack_epochs, patience=6)
        post_df = post_df.rename(columns={"raw_top1": f"post_top1_encseed{enc_seed}"})
        per_encoder_seed_post.append(post_df[["user_id", f"post_top1_encseed{enc_seed}"]])
        print(f"{tag} encoder seed {enc_seed} done in {time.time() - t0:.1f}s", flush=True)

    merged = feats.merge(per_encoder_seed_post[0][["user_id"]], on="user_id", how="inner")
    for frame in per_encoder_seed_post:
        merged = merged.merge(frame, on="user_id", how="left")
    seed_cols = [f"post_top1_encseed{s}" for s in encoder_seeds]
    merged["post_top1"] = merged[seed_cols].mean(axis=1)
    merged["post_top1_std"] = merged[seed_cols].std(axis=1)
    merged["raw_top1"] = merged["user_id"].map(raw_lookup).fillna(0.0)
    n_total = len(feats)
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")

    test_counts = pd.Series(original_windows["test"]["user"]).value_counts()
    merged["n_windows"] = merged["user_id"].map(lambda u: int(test_counts.get(u - 1, 0)))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "per_user_04d.csv", index=False)

    result = dict(
        dataset=args.dataset, mechanism="trajgan",
        encoder_seeds=list(encoder_seeds), attacker_seeds=list(attacker_seeds),
        max_epochs=args.max_epochs, patience=args.patience, min_epochs=args.min_epochs, val_frac=args.val_frac,
        trajgan_epochs=args.trajgan_epochs, trajgan_seed=args.trajgan_seed, trajgan_summary=trajgan_summary,
        git_commit=_git_commit(),
        actual_epochs_per_encoder_seed=actual_epochs_list,
        utility_mean=float(np.mean(utilities)), utility_std=float(np.std(utilities)),
        utility_per_encoder_seed=utilities,
        n_users_before_filter=n_total, n_users_after_filter=len(merged),
        output_vocab_size=original_total_locations, embedding_vocab_size=embedding_vocab,
        attacker_internal_split_sizes=dict(train=len(atk_train_idx), test=len(atk_test_idx)),
    )
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
