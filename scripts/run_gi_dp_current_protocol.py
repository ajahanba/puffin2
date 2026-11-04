#!/usr/bin/env python
"""GI-DP baseline, current protocol, per the PI's instruction (2026-11-03/04).
Replaces the retired adaptive-attacker-standard wrapper
(scripts/run_phase5b_gap_mechanism.py). One (dataset, epsilon) per
invocation; 3 encoder seeds.

Utility: next-location Acc@1 of the SAME predictor architecture Mo-PAE
uses (FixedEncoder + next_predictor MLPHead, puffin.models.mopae_torch),
trained on the PERTURBED release's windows but evaluated against the
TRUE (unperturbed) next-location labels -- GI-DP has no reconstruction
or privacy-adversary term, so only the encoder+next_predictor pair
trains, via plain cross-entropy, with the same early-stopping protocol
(chronological val split, patience) as everywhere else this round.

Risk: the SAME strongest-per-user external attacker
(train_raw_attacker_strongest_per_user), trained AND tested entirely on
the perturbed release (its own internal train/test split).

Output: a standard per_user_04d.csv (entropy_decile from TRAIN-split
ORIGINAL data, raw_top1 reused from the dataset's existing no_privacy
run since raw risk does not depend on the mechanism, post_top1 from the
attacker on the perturbed release) and summary.json, same shape as
run_phase4d_approved_batch.py's output so every downstream analysis
script (compute_individual_fairness.py, the plateau audit, etc.) works
unchanged.
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
from puffin.baselines.gi_dp import run_gi_dp
from puffin.data.tessellation import assign_grid_cells
from puffin.data.splits import temporal_split
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_strongest_per_user, _chronological_val_split
from puffin.models.mopae_torch import FixedEncoder, MLPHead


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def train_predictor(perturbed_tr, true_next_cell_tr, total_locations, device, seed,
                     max_epochs=80, patience=8, min_epochs=20, val_frac=0.1, batch_size=512, lr=3e-4):
    torch.manual_seed(seed)
    sl = perturbed_tr["cell"].shape[1]
    encoder = FixedEncoder(sl, total_locations).to(device)
    head = MLPHead(total_locations).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    n = len(perturbed_tr["user"])
    rng = np.random.default_rng(seed)
    if "ts" in perturbed_tr:
        val_idx_np, fit_idx_np = _chronological_val_split(perturbed_tr["ts"], perturbed_tr["user"], val_frac, rng)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(val_frac * n))
        val_idx_np, fit_idx_np = perm[:n_val], perm[n_val:]

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr = to_t(perturbed_tr["cell"]), to_t(perturbed_tr["weekday"]), to_t(perturbed_tr["hour"])
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


def eval_utility(encoder, head, perturbed_te, true_next_cell_te, device, batch_size=1024):
    encoder.eval(); head.eval()
    n = len(perturbed_te["user"])
    correct = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl_ = slice(start, start + batch_size)
            cell = torch.tensor(perturbed_te["cell"][sl_], dtype=torch.long, device=device)
            wd = torch.tensor(perturbed_te["weekday"][sl_], dtype=torch.long, device=device)
            hr = torch.tensor(perturbed_te["hour"][sl_], dtype=torch.long, device=device)
            y = torch.tensor(true_next_cell_te[sl_], dtype=torch.long, device=device)
            logits = head(encoder(cell, wd, hr))
            correct += (logits.argmax(-1) == y).sum().item()
    return correct / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--epsilon", type=float, required=True)
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
    ap.add_argument("--encoder-seeds", type=str, default="0,1,2")
    ap.add_argument("--attacker-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    encoder_seeds = tuple(int(s) for s in args.encoder_seeds.split(","))
    attacker_seeds = tuple(int(s) for s in args.attacker_seeds.split(","))
    tag = f"[{args.dataset} gi_dp eps={args.epsilon}]"

    df = pd.read_parquet(args.parquet)
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    total_users = int(df["user_id"].max())
    df = assign_grid_cells(df, resolution_deg=0.01)
    original_mapping = df.attrs["grid_cell_mapping"]
    total_locations = max(original_mapping.values())
    print(f"{tag} total_locations={total_locations}, building ORIGINAL windows...", flush=True)

    original_windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                         respect_traj_boundary=False)

    print(f"{tag} perturbing with epsilon={args.epsilon}...", flush=True)
    perturbed_df = run_gi_dp(df, epsilon=args.epsilon, seed=0, cell_mapping=original_mapping)
    # The perturbation can push points into physical cells never visited in
    # the original data (more likely at large epsilon); assign_grid_cells
    # extends the reused mapping with NEW ids beyond the original's max for
    # those, so the embedding layer must be sized for whatever the
    # PERTURBED release actually uses, not just the original's footprint.
    total_locations = max(total_locations, int(perturbed_df["cell_id"].max()))
    print(f"{tag} total_locations after perturbation: {total_locations}", flush=True)
    perturbed_windows = build_sl_windows(perturbed_df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                         respect_traj_boundary=False)

    for split in ("train", "test"):
        assert np.array_equal(original_windows[split]["user"], perturbed_windows[split]["user"]), \
            f"{tag} {split}: window alignment broke between original and perturbed -- cannot pair labels"
        assert np.array_equal(original_windows[split]["ts"], perturbed_windows[split]["ts"]), \
            f"{tag} {split}: window alignment broke (ts)"
    print(f"{tag} window alignment verified ({len(original_windows['train']['user'])} train, "
          f"{len(original_windows['test']['user'])} test)", flush=True)

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
        print(f"{tag} encoder seed {enc_seed}: training predictor...", flush=True)
        encoder, head, actual_epochs = train_predictor(
            perturbed_windows["train"], original_windows["train"]["next_cell"], total_locations, device, enc_seed,
            max_epochs=args.max_epochs, patience=args.patience, min_epochs=args.min_epochs, val_frac=args.val_frac)
        utility = eval_utility(encoder, head, perturbed_windows["test"], original_windows["test"]["next_cell"], device)
        utilities.append(utility)
        actual_epochs_list.append(actual_epochs)
        print(f"{tag} encoder seed {enc_seed}: stopped after {actual_epochs} epochs, utility={utility:.4f}, "
              f"post attacker ({len(attacker_seeds)} seeds) on perturbed release...", flush=True)

        post_df, _ = train_raw_attacker_strongest_per_user(
            perturbed_windows, total_locations, total_users, device, seeds=attacker_seeds,
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
    user_id_map = {u - 1: u for u in merged["user_id"]}  # 0-indexed window user -> 1-indexed user_id
    merged["n_windows"] = merged["user_id"].map(lambda u: int(test_counts.get(u - 1, 0)))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "per_user_04d.csv", index=False)

    result = dict(
        dataset=args.dataset, mechanism="gi_dp", epsilon=args.epsilon,
        encoder_seeds=list(encoder_seeds), attacker_seeds=list(attacker_seeds),
        max_epochs=args.max_epochs, patience=args.patience, min_epochs=args.min_epochs, val_frac=args.val_frac,
        git_commit=_git_commit(),
        actual_epochs_per_encoder_seed=actual_epochs_list,
        utility_mean=float(np.mean(utilities)), utility_std=float(np.std(utilities)),
        utility_per_encoder_seed=utilities,
        n_users_before_filter=n_total, n_users_after_filter=len(merged),
    )
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
