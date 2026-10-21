#!/usr/bin/env python
"""Checkpoint 04d: the PI-approved batch closing 04c's gaps.

mechanism in {plain, ed_soft, no_privacy}:
  plain       -- lambda4=0, any fairness_fn (inert), the missing
                 utility-logged plain-Mo-PAE reference 04c needed.
  ed_soft     -- the old PUFFIN L_pair_threshold baseline, re-run with
                 utility logging (04c excluded it entirely for lacking
                 this).
  no_privacy  -- the new utility ceiling: lambda1=lambda3=lambda4=0
                 (prediction loss only, no reconstruction, no privacy
                 adversary, no fairness term) -- what utility looks like
                 with no privacy pressure at all.
  floor       -- GeoLife's fixed-floor_loss rerun with the chance-
                 relative fix (puffin.fairness.losses.floor_loss).

Uses >=3 ENCODER-training seeds per (dataset, mechanism, lambda4), each
attacked with the usual 5 attacker seeds -- per the PI's instruction
that comparison runs report means/CIs over encoder seeds as well as
attacker seeds, not just attacker seeds (every other Phase 4 point used
exactly one encoder seed). Also switches to the PI's new primary
standard for THIS batch: the hardened external attacker, strongest per
user across MLP/GRU/GBT (puffin.attacks.reid's *_strongest_per_user
functions), not the older best-of-seed-only `_hardened` functions the
original Phase 4 grid used.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_strongest_per_user, train_attacker_on_frozen_encoder_strongest_per_user
from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.fairness.losses import ed_soft, floor_loss, group_dro_loss, build_entropy_table
from puffin.fairness.attacker_in_the_loop import refit_fresh_attacker_head, attacker_in_the_loop_train_step
from puffin.models.mopae_torch import build_bundle, fixed_train_step


def train_attacker_in_the_loop_mechanism(windows, total_locations, total_users, entropy_table, lambda4,
                                          loss_variant, refit_every_k_epochs, device, epochs,
                                          lambda_priv=0.3, batch_size=512, lr=3e-4, seed=0,
                                          head_refit_epochs=10, head_refit_lr=1e-3):
    """mechanism="attacker_in_the_loop": every `refit_every_k_epochs`, a
    FRESH attacker head is refit to convergence on the momentarily-
    frozen encoder, then used (frozen) as the fairness term's signal
    source for the next k epochs. loss_variant in {"floor", "group_dro"}."""
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
    loss_fn = floor_loss if loss_variant == "floor" else group_dro_loss

    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)
    fresh_head = None
    for epoch in range(epochs):
        if epoch % refit_every_k_epochs == 0:
            fresh_head = refit_fresh_attacker_head(bundle.encoder, cell, wd, hr, user_idx, total_users,
                                                     device, epochs=head_refit_epochs, lr=head_refit_lr)
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            attacker_in_the_loop_train_step(bundle, opts, fresh_head, cell[idx], wd[idx], hr[idx],
                                             y_next_idx[idx], user_idx[idx], coord_target[idx],
                                             loss_fn, entropy_table, lambda4, lambdas)
    return bundle


def train_mechanism(windows, total_locations, total_users, entropy_table, mechanism, lambda4, device,
                     epochs, lambda_priv=0.3, batch_size=512, lr=3e-4, seed=0,
                     loss_variant=None, refit_every_k_epochs=None):
    if mechanism == "attacker_in_the_loop":
        return train_attacker_in_the_loop_mechanism(
            windows, total_locations, total_users, entropy_table, lambda4, loss_variant,
            refit_every_k_epochs, device, epochs, lambda_priv=lambda_priv, batch_size=batch_size,
            lr=lr, seed=seed)

    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    if mechanism == "no_privacy":
        lambdas = dict(l_recon=0.0, l_util=1.0, l_priv=0.0)
        fairness_fn, l4 = None, 0.0
    elif mechanism == "plain":
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = None, 0.0
    elif mechanism == "ed_soft":
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = ed_soft, lambda4
    elif mechanism == "floor":
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = floor_loss, lambda4
    else:
        raise ValueError(mechanism)

    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                              coord_target[idx], fairness_fn=fairness_fn, entropy_table=entropy_table,
                              lambda4=l4, lambdas=lambdas)
    return bundle


def eval_in_model_confidence(bundle, windows, device):
    """The CO-TRAINED (in-model) adversary's own per-user confidence on
    test windows -- the quantity any floor/group-DRO-style fairness loss
    actually optimizes against, as distinct from a fresh EXTERNAL
    attacker's accuracy on the same frozen features. Reported by decile
    so it can sit next to the external attacker's top-1 in the report:
    per the PI, this is the mechanistic explanation for why loss-based
    fairness formulations can't target real risk -- the in-model
    adversary the encoder actually trains against stays near chance
    throughout training (confirmed via scripts/diagnose_floor_loss.py),
    while a real external attacker on the same features does far
    better."""
    te = windows["test"]
    cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(te["user"], dtype=torch.long, device=device)
    bundle.encoder.eval()
    bundle.id_recognizer.eval()
    with torch.no_grad():
        enc = bundle.encoder(cell, wd, hr)
        id_logits = bundle.id_recognizer(enc)
        probs = torch.softmax(id_logits, dim=-1)
        true_onehot = torch.nn.functional.one_hot(user_idx, num_classes=id_logits.shape[-1]).float()
        confidence = (true_onehot * probs).sum(-1).cpu().numpy()
    bundle.encoder.train()
    bundle.id_recognizer.train()
    user_1idx = te["user"] + 1
    return pd.DataFrame(dict(user_id=user_1idx, confidence=confidence)).groupby(
        "user_id")["confidence"].mean().reset_index()


def eval_utility(bundle, windows, device):
    te = windows["test"]
    cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(te["next_cell"], dtype=torch.long, device=device)
    bundle.encoder.eval()
    bundle.next_predictor.eval()
    with torch.no_grad():
        enc = bundle.encoder(cell, wd, hr)
        preds = bundle.next_predictor(enc).argmax(dim=-1)
        acc = (preds == y_next_idx).float().mean().item()
    bundle.encoder.train()
    bundle.next_predictor.train()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mechanism", required=True,
                     choices=["plain", "ed_soft", "no_privacy", "floor", "attacker_in_the_loop"])
    ap.add_argument("--loss-variant", choices=["floor", "group_dro"], default=None,
                     help="attacker_in_the_loop only")
    ap.add_argument("--refit-every-k-epochs", type=int, default=None, help="attacker_in_the_loop only")
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--mopae-epochs", type=int, default=20)
    ap.add_argument("--lambda-priv", type=float, default=0.3)
    ap.add_argument("--lambda4", type=float, default=0.0)
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--encoder-seeds", type=str, default="0,1,2")
    ap.add_argument("--attacker-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    encoder_seeds = tuple(int(s) for s in args.encoder_seeds.split(","))
    attacker_seeds = tuple(int(s) for s in args.attacker_seeds.split(","))
    tag = f"{args.dataset} {args.mechanism} lambda4={args.lambda4}"

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

    print(f"[{tag}] building windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)

    print(f"[{tag}] raw attacker (strongest per user, MLP+GRU+GBT, {len(attacker_seeds)} seeds)...", flush=True)
    raw_df, _ = train_raw_attacker_strongest_per_user(windows, total_locations, total_users, device,
                                                       seeds=attacker_seeds, max_epochs=args.attack_epochs,
                                                       patience=6)

    per_encoder_seed = []
    per_encoder_seed_inmodel = []
    utilities = []
    train_wall_times = []
    for enc_seed in encoder_seeds:
        print(f"[{tag}] encoder seed {enc_seed}: training mechanism...", flush=True)
        _t0 = time.time()
        bundle = train_mechanism(windows, total_locations, total_users, entropy_table, args.mechanism,
                                  args.lambda4, device, epochs=args.mopae_epochs, lambda_priv=args.lambda_priv,
                                  seed=enc_seed, loss_variant=args.loss_variant,
                                  refit_every_k_epochs=args.refit_every_k_epochs)
        train_wall_times.append(time.time() - _t0)
        utility = eval_utility(bundle, windows, device)
        utilities.append(utility)
        print(f"[{tag}] encoder seed {enc_seed}: utility={utility:.4f}, "
              f"post attacker ({len(attacker_seeds)} attacker seeds)...", flush=True)
        post_df, _ = train_attacker_on_frozen_encoder_strongest_per_user(
            bundle.encoder, windows, total_users, device, seeds=attacker_seeds,
            max_epochs=args.attack_epochs, patience=6)
        post_df = post_df.rename(columns={"post_top1": f"post_top1_encseed{enc_seed}"})
        per_encoder_seed.append(post_df[["user_id", f"post_top1_encseed{enc_seed}"]])

        inmodel_df = eval_in_model_confidence(bundle, windows, device)
        inmodel_df = inmodel_df.rename(columns={"confidence": f"inmodel_confidence_encseed{enc_seed}"})
        per_encoder_seed_inmodel.append(inmodel_df[["user_id", f"inmodel_confidence_encseed{enc_seed}"]])

    merged = feats.merge(raw_df[["user_id", "n_windows", "raw_top1"]], on="user_id", how="inner")
    for df_ in per_encoder_seed:
        merged = merged.merge(df_, on="user_id", how="inner")
    for df_ in per_encoder_seed_inmodel:
        merged = merged.merge(df_, on="user_id", how="left")
    enc_cols = [c for c in merged.columns if c.startswith("post_top1_encseed")]
    inmodel_cols = [c for c in merged.columns if c.startswith("inmodel_confidence_encseed")]
    merged["post_top1"] = merged[enc_cols].mean(axis=1)
    merged["post_top1_std_over_encoder_seeds"] = merged[enc_cols].std(axis=1)
    merged["inmodel_confidence"] = merged[inmodel_cols].mean(axis=1)
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")

    filtered = merged[merged["n_windows"] >= 10].copy()
    n_excluded = len(merged) - len(filtered)

    deciles = filtered["entropy_decile"].to_numpy()
    raw_ci = bootstrap_decile_ci(filtered["raw_top1"].to_numpy(), deciles)
    post_ci = bootstrap_decile_ci(filtered["post_top1"].to_numpy(), deciles)
    inmodel_ci = bootstrap_decile_ci(filtered["inmodel_confidence"].to_numpy(), deciles)
    chance_level = 1.0 / total_users

    result = dict(
        dataset=args.dataset, mechanism=args.mechanism, lambda4=args.lambda4,
        loss_variant=args.loss_variant, refit_every_k_epochs=args.refit_every_k_epochs,
        encoder_seeds=list(encoder_seeds), attacker_seeds=list(attacker_seeds),
        utility_mean=float(np.mean(utilities)), utility_std=float(np.std(utilities)),
        utility_per_encoder_seed=utilities,
        train_wall_time_seconds_per_encoder_seed=train_wall_times,
        train_wall_time_seconds_mean=float(np.mean(train_wall_times)),
        n_users_before_filter=len(merged), n_users_after_filter=len(filtered), n_excluded=n_excluded,
        chance_level=chance_level,
        raw_decile_ci=raw_ci, post_decile_ci=post_ci, inmodel_confidence_decile_ci=inmodel_ci,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_dir / "per_user_04d.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("raw_decile_ci", "post_decile_ci", "inmodel_confidence_decile_ci")}, indent=2))


if __name__ == "__main__":
    main()
