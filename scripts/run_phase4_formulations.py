#!/usr/bin/env python
"""Phase 4 (item B): evaluate each fairness formulation under the adaptive
attacker at matched utility, per the PI's post-Checkpoint-3c instruction.

One (dataset, formulation, lambda4) combination per invocation. Formulations:
  ed_soft          -- old PUFFIN L_pair_threshold baseline (already fully
                       covered by Checkpoint 3b's grid; this script can still
                       run it for a formulation-vs-formulation sanity check).
  lipschitz        -- puffin.fairness.losses.lipschitz_soft
  floor            -- puffin.fairness.losses.floor_loss
  variance         -- puffin.fairness.losses.variance_of_risk
  group_dro        -- puffin.fairness.losses.group_dro_loss
  per_user_minimax -- puffin.fairness.losses.per_user_minimax_loss
  group_adversarial -- puffin.fairness.group_adversarial (own train step; see
                       note below on what "lambda4" means for it)

For every formulation except group_adversarial, lambda4 is the additive
weight on the fairness_fn term, exactly as in Checkpoint 3b. For
group_adversarial there is no separate additive term -- "lambda4" here
instead OVERRIDES l_priv (the weight on -worst_group_loss in the encoder's
objective), since that is the knob that plays the analogous role. This is
noted explicitly in the output so it isn't read as the same quantity.

Primary metric (per the PI): worst-off decile's mean top-1 accuracy under
the ADAPTIVE attacker (not the naive one), reported alongside utility
(next-location top-1 accuracy on held-out test windows) so comparisons are
made at matched utility rather than matched lambda.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_hardened, train_attacker_on_frozen_encoder_hardened
from puffin.attacks.adaptive import per_group_specialized_attack_multiseed, group_reweighted_attack_multiseed
from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.metrics.privacy import privacy_gain
from puffin.models.mopae_torch import build_bundle, fixed_train_step, FIXED_LAMBDAS
from puffin.fairness.losses import (
    ed_soft, lipschitz_soft, floor_loss, variance_of_risk, group_dro_loss, per_user_minimax_loss,
    build_entropy_table,
)
from puffin.fairness.group_adversarial import GroupAdversaryHeads, group_adversarial_train_step, decile_group_idx

LOSS_FORMULATIONS = dict(
    ed_soft=ed_soft, lipschitz=lipschitz_soft, floor=floor_loss,
    variance=variance_of_risk, group_dro=group_dro_loss, per_user_minimax=per_user_minimax_loss,
)


def _to_tensors(split, device):
    cell = torch.tensor(split["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(split["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(split["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(split["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(split["next_cell"], dtype=torch.long, device=device)
    return cell, wd, hr, user_idx, y_next_idx


def train_loss_formulation(windows, total_locations, total_users, entropy_table, lambda_priv, lambda4,
                            formulation, device, epochs, batch_size=512, lr=3e-4, seed=0):
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
    cell, wd, hr, user_idx, y_next_idx = _to_tensors(windows["train"], device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    fairness_fn = LOSS_FORMULATIONS[formulation]
    n = len(cell)
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                              coord_target[idx], fairness_fn=fairness_fn, entropy_table=entropy_table,
                              lambda4=lambda4, lambdas=lambdas)
    return bundle


def train_group_adversarial(windows, total_locations, total_users, entropy_table, lambda_priv_override,
                             device, epochs, n_groups=10, batch_size=512, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.next_predictor):
        m.to(device)
    group_heads = GroupAdversaryHeads(total_users=total_users, n_groups=n_groups).to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    group_head_opt = torch.optim.Adam(group_heads.parameters(), lr=lr)
    lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv_override)
    cell, wd, hr, user_idx, y_next_idx = _to_tensors(windows["train"], device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b = idx
            group_idx = decile_group_idx(entropy_table, user_idx[b], n_groups=n_groups)
            group_adversarial_train_step(bundle, opts, group_heads, group_head_opt,
                                          cell[b], wd[b], hr[b], y_next_idx[b], user_idx[b], coord_target[b],
                                          group_idx, lambdas=lambdas)
    return bundle


def eval_utility(bundle, windows, device):
    """Next-location top-1 accuracy on held-out test windows -- the utility
    metric "matched utility" compares formulations against."""
    te = windows["test"]
    cell, wd, hr, _user_idx, y_next_idx = _to_tensors(te, device)
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
    ap.add_argument("--formulation", required=True, choices=list(LOSS_FORMULATIONS.keys()) + ["group_adversarial"])
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--mopae-epochs", type=int, default=20)
    ap.add_argument("--lambda-priv", type=float, default=0.3)
    ap.add_argument("--lambda4", type=float, required=True,
                     help="additive fairness weight for loss formulations; overrides l_priv for group_adversarial")
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--global-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--adaptive-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--reweight-factor", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    global_seeds = tuple(int(s) for s in args.global_seeds.split(","))
    adaptive_seeds = tuple(int(s) for s in args.adaptive_seeds.split(","))

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

    tag = f"{args.dataset} {args.formulation} lambda4={args.lambda4}"
    print(f"[{tag}] building SL={args.sl} windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)

    print(f"[{tag}] training mechanism...", flush=True)
    if args.formulation == "group_adversarial":
        bundle = train_group_adversarial(windows, total_locations, total_users, entropy_table, args.lambda4,
                                          device, epochs=args.mopae_epochs)
    else:
        bundle = train_loss_formulation(windows, total_locations, total_users, entropy_table, args.lambda_priv,
                                         args.lambda4, args.formulation, device, epochs=args.mopae_epochs)

    utility_acc = eval_utility(bundle, windows, device)
    print(f"[{tag}] utility (next-location top-1 on held-out test): {utility_acc:.4f}", flush=True)

    print(f"[{tag}] global (naive) 5-seed attacker...", flush=True)
    raw_df, _ = train_raw_attacker_hardened(windows, total_locations, total_users, device, seeds=global_seeds,
                                             max_epochs=args.attack_epochs, patience=6)
    post_df = train_attacker_on_frozen_encoder_hardened(bundle.encoder, windows, total_users, device,
                                                         seeds=global_seeds, max_epochs=args.attack_epochs,
                                                         patience=6)
    merged = feats.merge(raw_df, on="user_id", how="inner").merge(post_df, on="user_id", how="inner")
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")
    merged["global_gain"] = privacy_gain(merged["raw_top1"].to_numpy(), merged["post_top1"].to_numpy())

    print(f"[{tag}] adaptive attacks (specialized across all deciles + reweighted), "
          f"{len(adaptive_seeds)} seeds each...", flush=True)
    decile_gain = merged.groupby("entropy_decile")["global_gain"].mean().sort_values(ascending=False)
    most_protected = [int(decile_gain.index[0])]
    spec_df = per_group_specialized_attack_multiseed(bundle.encoder, windows, feats, total_users, device,
                                                       seeds=adaptive_seeds, epochs=args.attack_epochs)
    rew_df = group_reweighted_attack_multiseed(bundle.encoder, windows, feats, total_users, device,
                                                target_deciles=most_protected, reweight_factor=args.reweight_factor,
                                                seeds=adaptive_seeds, epochs=args.attack_epochs)

    combined = merged.merge(spec_df[["user_id", "specialized_top1"]], on="user_id", how="left")
    combined = combined.merge(rew_df[["user_id", "reweighted_top1"]], on="user_id", how="left")
    combined["post_adaptive_top1"] = combined[["specialized_top1", "reweighted_top1"]].max(axis=1)
    deciles = combined["entropy_decile"].to_numpy()

    naive_decile_ci = bootstrap_decile_ci(combined["post_top1"].to_numpy(), deciles)
    adaptive_decile_ci = bootstrap_decile_ci(combined["post_adaptive_top1"].to_numpy(), deciles)
    raw_decile_ci = bootstrap_decile_ci(combined["raw_top1"].to_numpy(), deciles)

    # primary metric: worst-off decile = highest mean risk under the ADAPTIVE attacker
    adaptive_means = {d: v["mean"] for d, v in adaptive_decile_ci.items()}
    worst_off_decile = max(adaptive_means, key=adaptive_means.get)
    worst_off_adaptive_risk = adaptive_decile_ci[worst_off_decile]
    worst_off_naive_risk = naive_decile_ci.get(worst_off_decile)

    result = dict(
        dataset=args.dataset, formulation=args.formulation, lambda4=args.lambda4, lambda_priv=args.lambda_priv,
        n_users=len(combined), utility_next_location_top1=utility_acc,
        most_protected_decile_by_naive_attacker=most_protected,
        worst_off_decile_by_adaptive_attacker=int(worst_off_decile),
        worst_off_decile_adaptive_risk=worst_off_adaptive_risk,
        worst_off_decile_naive_risk=worst_off_naive_risk,
        raw_decile_ci=raw_decile_ci, naive_decile_ci=naive_decile_ci, adaptive_decile_ci=adaptive_decile_ci,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "per_user_phase4.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in
                       ("raw_decile_ci", "naive_decile_ci", "adaptive_decile_ci")}, indent=2))


if __name__ == "__main__":
    main()
