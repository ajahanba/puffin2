#!/usr/bin/env python
"""Checkpoint 2c: robustness checks the PI asked for before Phase 3 goes deep.

  1. GeoLife negative-risk-reduction: pair raw/post attacker results BY
     SEED (not by comparing two independently-averaged means), bootstrap
     95% CI over users for each decile's absolute risk reduction, state
     plainly whether the low deciles' CI excludes zero.
  2. Regression at (user, seed) granularity with user-clustered SEs;
     coefficients + 95% CIs, not just p-values.
  3. Attacker gap: chronological (not random) per-user validation split for
     early stopping (puffin.attacks.reid's default now); disparate risk
     under (a) a contemporaneous split (no time gap between what the
     attacker trains on and what it's evaluated against) vs (b) the
     existing future/temporal split. Does the entropy gap hold under (a)?
     Does drift (future acc - contemporaneous acc, per user) vary by decile?
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
from puffin.attacks.windowing import build_sl_windows, build_sl_windows_contemporaneous
from puffin.attacks.reid import train_raw_attacker_hardened, train_attacker_on_frozen_encoder_hardened
from puffin.attacks.analysis import ols_cluster_robust, bootstrap_decile_ci
from scripts.run_phase2_disparate_risk import train_mopae_plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--lambda-priv", type=float, default=0.3, help="the meaningful operating point from 2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = tuple(int(s) for s in args.seeds.split(","))

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = dict(dataset=args.dataset)

    # ---- Item 1 & 2 setup: future (temporal) split, chronological val, raw + post attacker ----
    print(f"[{args.dataset}] building FUTURE (temporal) split windows...", flush=True)
    windows_future = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                       respect_traj_boundary=args.respect_traj_boundary)
    print(f"[{args.dataset}] future: train={len(windows_future['train']['user'])} "
          f"test={len(windows_future['test']['user'])}", flush=True)

    print(f"[{args.dataset}] raw attacker, future split, chronological val...", flush=True)
    raw_df, raw_winners = train_raw_attacker_hardened(
        windows_future, total_locations, total_users, device, seeds=seeds,
        max_epochs=args.max_epochs, patience=args.patience, chronological_val=True)

    print(f"[{args.dataset}] plain Mo-PAE (lambda4=0) at lambda_priv={args.lambda_priv}...", flush=True)
    bundle = train_mopae_plain(windows_future, total_locations, total_users, args.lambda_priv, device,
                                epochs=args.max_epochs // 3 or 1)
    print(f"[{args.dataset}] frozen-encoder attacker, future split, chronological val...", flush=True)
    post_df = train_attacker_on_frozen_encoder_hardened(
        bundle.encoder, windows_future, total_users, device, seeds=seeds,
        max_epochs=args.max_epochs, patience=args.patience, chronological_val=True)

    merged = feats.merge(raw_df, on="user_id", how="inner").merge(post_df, on="user_id", how="inner")
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")

    # ---- Item 1: seed-paired absolute reduction, bootstrap CI, explicit "excludes zero" ----
    for s in seeds:
        merged[f"abs_reduction_seed{s}"] = merged[f"top1_seed{s}"] - merged[f"post_top1_seed{s}"]
    seed_cols = [f"abs_reduction_seed{s}" for s in seeds]
    merged["paired_abs_reduction"] = merged[seed_cols].mean(axis=1)  # seed-paired, THEN averaged over seeds
    paired_ci = bootstrap_decile_ci(merged["paired_abs_reduction"].to_numpy(), merged["entropy_decile"].to_numpy())
    result["item1_seed_paired_abs_reduction_ci"] = paired_ci
    result["item1_low_deciles_exclude_zero"] = {
        d: paired_ci[d]["ci_excludes_zero"] for d in paired_ci if d <= 2
    }

    # ---- Item 2: (user, seed)-level regression, cluster-robust SEs ----
    rows = []
    for s in seeds:
        tmp = merged[["user_id", "s_unc", "rog_km", "n_windows", "n_distinct_cells", f"top1_seed{s}"]].copy()
        tmp = tmp.rename(columns={f"top1_seed{s}": "risk"})
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    ols = ols_cluster_robust(long_df["risk"].to_numpy(), dict(
        s_unc=long_df["s_unc"].to_numpy(), rog_km=long_df["rog_km"].to_numpy(),
        n_windows=long_df["n_windows"].to_numpy(), n_distinct_cells=long_df["n_distinct_cells"].to_numpy()),
        long_df["user_id"].to_numpy())
    result["item2_ols_cluster_robust_user_seed_level"] = ols

    # ---- Item 3: contemporaneous split, same pipeline, random (not chronological) val ----
    print(f"[{args.dataset}] building CONTEMPORANEOUS split windows...", flush=True)
    windows_contemp = build_sl_windows_contemporaneous(
        df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
        respect_traj_boundary=args.respect_traj_boundary)
    print(f"[{args.dataset}] contemporaneous: train={len(windows_contemp['train']['user'])} "
          f"test={len(windows_contemp['test']['user'])}", flush=True)

    print(f"[{args.dataset}] raw attacker, contemporaneous split, random val (no 'future' to be chronological about)...",
          flush=True)
    raw_contemp_df, contemp_winners = train_raw_attacker_hardened(
        windows_contemp, total_locations, total_users, device, seeds=seeds,
        max_epochs=args.max_epochs, patience=args.patience, chronological_val=False)

    contemp_merged = feats.merge(raw_contemp_df, on="user_id", how="inner")
    contemp_merged["entropy_decile"] = pd.qcut(contemp_merged["s_unc"], 10, labels=False, duplicates="drop")
    contemp_decile_ci = bootstrap_decile_ci(contemp_merged["raw_top1"].to_numpy(),
                                             contemp_merged["entropy_decile"].to_numpy())
    result["item3_contemporaneous_decile_risk_ci"] = contemp_decile_ci
    result["item3_future_decile_risk_ci"] = bootstrap_decile_ci(
        merged["raw_top1"].to_numpy(), merged["entropy_decile"].to_numpy())
    result["item3_contemporaneous_mean_top1"] = float(contemp_merged["raw_top1"].mean())
    result["item3_future_mean_top1"] = float(merged["raw_top1"].mean())
    result["item3_contemporaneous_winners"] = contemp_winners
    result["item3_future_winners"] = raw_winners

    drift = merged[["user_id", "raw_top1", "entropy_decile"]].merge(
        contemp_merged[["user_id", "raw_top1"]].rename(columns={"raw_top1": "contemp_top1"}),
        on="user_id", how="inner")
    drift["drift"] = drift["raw_top1"] - drift["contemp_top1"]  # future - contemporaneous; negative = got harder over time
    drift_ci = bootstrap_decile_ci(drift["drift"].to_numpy(), drift["entropy_decile"].to_numpy())
    result["item3_drift_by_decile_ci"] = drift_ci
    result["item3_n_users_in_drift_comparison"] = len(drift)

    merged.to_csv(out_dir / "per_user_future.csv", index=False)
    contemp_merged.to_csv(out_dir / "per_user_contemporaneous.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
