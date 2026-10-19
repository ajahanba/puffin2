#!/usr/bin/env python
"""Checkpoint 3b: harden the Phase 3 task 5 adaptive-adversary result.

One (dataset, lambda4) combination per invocation -- meant to be run as a
grid (lambda4 in {0.0, 0.3, 1.0, 3.0} x {geolife, foursquare, gowalla_nyc})
and compared across runs' summary.json files, not all in one process.

For the given mechanism (lambda4=0.0 is PLAIN Mo-PAE, the PI's stop-
condition gate; >0 is fairness-active):
  1. Train the mechanism, find its most-protected decile via the ordinary
     5-seed global attacker (puffin.attacks.reid).
  2. Run BOTH adaptive variants with 5 SEEDS and bootstrap CIs
     (puffin.attacks.adaptive's *_multiseed functions + bootstrap_decile_ci).
  3. Report the 3-stage per-decile table: raw risk -> post-mechanism risk
     under a NAIVE attacker -> post-mechanism risk under the ADAPTIVE
     attacker.
  4. Run the threat-model check once (can_infer_decile_from_features): can
     the specialized attacker's group knowledge come from released
     features alone, or is it side information?
  5. Run the in-house MIA across 5 seeds, report AUC mean + bootstrap CI.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_hardened, train_attacker_on_frozen_encoder_hardened
from puffin.attacks.adaptive import (
    user_decile_map, per_group_specialized_attack_multiseed, group_reweighted_attack_multiseed,
    can_infer_decile_from_features,
)
from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.attacks.mia_inhouse import run_loss_threshold_mia
from puffin.metrics.privacy import privacy_gain
from scripts.run_phase3_adaptive_adversary import train_mopae_fair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--mopae-epochs", type=int, default=20)
    ap.add_argument("--lambda-priv", type=float, default=0.3)
    ap.add_argument("--lambda4", type=float, required=True)
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--global-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--adaptive-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--reweight-factor", type=float, default=10.0)
    ap.add_argument("--run-threat-model-check", action="store_true")
    ap.add_argument("--run-mia", action="store_true")
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
    from puffin.fairness.losses import build_entropy_table
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

    print(f"[{args.dataset} lambda4={args.lambda4}] building SL={args.sl} windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)

    print(f"[{args.dataset} lambda4={args.lambda4}] training mechanism...", flush=True)
    bundle = train_mopae_fair(windows, total_locations, total_users, entropy_table,
                               args.lambda_priv, args.lambda4, device, epochs=args.mopae_epochs)

    print(f"[{args.dataset} lambda4={args.lambda4}] global (naive) 5-seed attacker...", flush=True)
    raw_df, _ = train_raw_attacker_hardened(windows, total_locations, total_users, device, seeds=global_seeds,
                                             max_epochs=args.attack_epochs, patience=6)
    post_df = train_attacker_on_frozen_encoder_hardened(bundle.encoder, windows, total_users, device,
                                                         seeds=global_seeds, max_epochs=args.attack_epochs,
                                                         patience=6)
    merged = feats.merge(raw_df, on="user_id", how="inner").merge(post_df, on="user_id", how="inner")
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")
    merged["global_gain"] = privacy_gain(merged["raw_top1"].to_numpy(), merged["post_top1"].to_numpy())
    decile_gain = merged.groupby("entropy_decile")["global_gain"].mean().sort_values(ascending=False)
    most_protected = [int(decile_gain.index[0])]
    print(f"[{args.dataset} lambda4={args.lambda4}] most-protected decile: {most_protected}", flush=True)

    print(f"[{args.dataset} lambda4={args.lambda4}] adaptive attacks, "
          f"{len(adaptive_seeds)} seeds each...", flush=True)
    spec_df = per_group_specialized_attack_multiseed(bundle.encoder, windows, feats, total_users, device,
                                                       seeds=adaptive_seeds, epochs=args.attack_epochs)
    rew_df = group_reweighted_attack_multiseed(bundle.encoder, windows, feats, total_users, device,
                                                target_deciles=most_protected, reweight_factor=args.reweight_factor,
                                                seeds=adaptive_seeds, epochs=args.attack_epochs)

    combined = merged.merge(spec_df[["user_id", "specialized_top1", "specialized_top1_std"]],
                             on="user_id", how="left")
    combined = combined.merge(rew_df[["user_id", "reweighted_top1", "reweighted_top1_std"]],
                               on="user_id", how="left")

    # 3-stage table: raw -> post (naive) -> post (adaptive, best of specialized/reweighted per user)
    combined["post_adaptive_top1"] = combined[["specialized_top1", "reweighted_top1"]].max(axis=1)
    deciles = combined["entropy_decile"].to_numpy()
    three_stage = {}
    for stage, col in [("raw", "raw_top1"), ("post_naive", "post_top1"), ("post_adaptive", "post_adaptive_top1")]:
        three_stage[stage] = bootstrap_decile_ci(combined[col].to_numpy(), deciles)

    target_rows = combined[combined["entropy_decile"].isin(most_protected)]
    target_group = dict(
        n_users=len(target_rows),
        mean_raw_top1=float(target_rows["raw_top1"].mean()),
        mean_post_naive_top1=float(target_rows["post_top1"].mean()),
        mean_post_specialized_top1=float(target_rows["specialized_top1"].mean()),
        mean_post_reweighted_top1=float(target_rows["reweighted_top1"].mean()),
        specialized_multiplier=(float(target_rows["specialized_top1"].mean() / target_rows["post_top1"].mean())
                                 if target_rows["post_top1"].mean() > 0 else float("nan")),
        reweighted_multiplier=(float(target_rows["reweighted_top1"].mean() / target_rows["post_top1"].mean())
                                if target_rows["post_top1"].mean() > 0 else float("nan")),
    )
    # bootstrap CI specifically on the target group's specialized/reweighted - naive delta
    spec_delta_ci = bootstrap_decile_ci(
        (combined["specialized_top1"] - combined["post_top1"]).to_numpy(), deciles)
    rew_delta_ci = bootstrap_decile_ci(
        (combined["reweighted_top1"] - combined["post_top1"]).to_numpy(), deciles)
    target_group["specialized_delta_ci"] = {d: spec_delta_ci[d] for d in most_protected if d in spec_delta_ci}
    target_group["reweighted_delta_ci"] = {d: rew_delta_ci[d] for d in most_protected if d in rew_delta_ci}

    result = dict(
        dataset=args.dataset, lambda4=args.lambda4, lambda_priv=args.lambda_priv,
        n_users=len(combined), most_protected_decile=most_protected,
        decile_mean_global_gain={int(k): float(v) for k, v in decile_gain.to_dict().items()},
        three_stage_decile_risk_ci=three_stage,
        target_group=target_group,
    )

    if args.run_threat_model_check:
        print(f"[{args.dataset} lambda4={args.lambda4}] threat-model check: "
              f"can decile be inferred from features alone?...", flush=True)
        result["threat_model_decile_inference"] = can_infer_decile_from_features(
            bundle.encoder, windows, feats, device, epochs=args.attack_epochs)

    if args.run_mia:
        print(f"[{args.dataset} lambda4={args.lambda4}] MIA across 5 seeds for a CI...", flush=True)
        aucs = []
        for s in adaptive_seeds:
            _, mia_summary = run_loss_threshold_mia(df, device, sl=args.sl,
                                                      max_windows_per_user=args.max_windows_per_user, seed=s,
                                                      epochs=args.attack_epochs,
                                                      respect_traj_boundary=args.respect_traj_boundary)
            aucs.append(mia_summary["overall_auc"])
        aucs = np.array([a for a in aucs if not np.isnan(a)])
        rng = np.random.default_rng(0)
        boots = np.array([aucs[rng.integers(0, len(aucs), size=len(aucs))].mean() for _ in range(2000)])
        result["mia_auc"] = dict(mean=float(aucs.mean()), ci_lo=float(np.quantile(boots, 0.025)),
                                  ci_hi=float(np.quantile(boots, 0.975)), per_seed=[float(a) for a in aucs])

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "per_user_3b.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
