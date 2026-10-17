#!/usr/bin/env python
"""Phase 2, hardened (post-Checkpoint-2 revision, PI-directed):

  1. Mobility features (puffin.features.mobility): S_rand, S_unc, S_real,
     radius of gyration, regularity, n_distinct_cells.
  2. Windowing coverage reported at SL=10 (main) and, for check-in datasets
     (--no-respect-traj-boundary), SL=5 (sensitivity) -- full-sequence
     windowing for check-in data (Foursquare/Gowalla), per-trajectory for
     GPS data (GeoLife); see puffin/attacks/windowing.py.
  3. A fresh, hardened raw re-id attacker: best of {MLP, GRU} per seed
     (chosen by validation accuracy), 5 seeds, early stopping. Reports which
     architecture produced every number.
  4. Per-user risk regressed on S_unc + radius_of_gyration + n_windows +
     n_distinct_cells, with proper standard errors/p-values -- does entropy
     survive the controls?
  5. Bootstrap 95% CIs over users for each entropy decile.
  6. PLAIN Mo-PAE (lambda4=0, no fairness term) at 2-3 operating points
     (varying lambda_priv), each followed by a hardened attacker on that
     run's frozen encoder output. Reports ABSOLUTE and RELATIVE risk
     reduction by decile, with bootstrap CIs. The lambda4 (fairness) sweep
     is deferred to Phases 4-5, per the PI.
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows, count_sl_coverage
from puffin.attacks.reid import train_raw_attacker_hardened, train_attacker_on_frozen_encoder_hardened
from puffin.attacks.analysis import ols_with_pvalues, bootstrap_decile_ci
from puffin.models.mopae_torch import build_bundle, fixed_train_step


def train_mopae_plain(windows: dict, total_locations: int, total_users: int, lambda_priv: float,
                       device, epochs: int, batch_size: int = 512, lr: float = 3e-4, seed: int = 0):
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
    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                              coord_target[idx], fairness_fn=None, lambda4=0.0, lambdas=lambdas)
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--sl-sensitivity", type=int, default=None, help="extra SL to report coverage/risk for")
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--lambda-priv-points", type=str, default="0.05,0.1,0.3")
    ap.add_argument("--skip-mopae", action="store_true",
                     help="raw-attacker + regression + bootstrap CI only; for a lighter SL sensitivity pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = tuple(int(s) for s in args.seeds.split(","))

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())

    print(f"[{args.dataset}] device={device}, respect_traj_boundary={args.respect_traj_boundary}", flush=True)
    print(f"[{args.dataset}] computing mobility features for {total_users} users...", flush=True)
    feats = compute_mobility_features(df)

    coverage = {args.sl: count_sl_coverage(df, args.sl, args.respect_traj_boundary)}
    if args.sl_sensitivity:
        coverage[args.sl_sensitivity] = count_sl_coverage(df, args.sl_sensitivity, args.respect_traj_boundary)
    print(f"[{args.dataset}] coverage: {json.dumps(coverage)}", flush=True)

    print(f"[{args.dataset}] building SL={args.sl} windows (respect_traj_boundary={args.respect_traj_boundary})...",
          flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)
    print(f"[{args.dataset}] train windows={len(windows['train']['user'])}, "
          f"test windows={len(windows['test']['user'])}", flush=True)

    print(f"[{args.dataset}] training hardened raw attacker "
          f"(best-of MLP/GRU, seeds={seeds}, early stopping)...", flush=True)
    raw_risk_df, winner_log = train_raw_attacker_hardened(
        windows, total_locations, total_users, device, seeds=seeds,
        max_epochs=args.max_epochs, patience=args.patience)
    print(f"[{args.dataset}] attacker winners by seed: {winner_log}", flush=True)

    merged = feats.merge(raw_risk_df, on="user_id", how="inner")
    print(f"[{args.dataset}] {len(merged)} users have both features and >=1 test window "
          f"(of {total_users} total, {len(feats)} with features)", flush=True)

    ols = ols_with_pvalues(merged["raw_top1"].to_numpy(), dict(
        s_unc=merged["s_unc"].to_numpy(), rog_km=merged["rog_km"].to_numpy(),
        n_windows=merged["n_windows"].to_numpy(), n_distinct_cells=merged["n_distinct_cells"].to_numpy()))

    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")
    decile_ci = bootstrap_decile_ci(merged["raw_top1"].to_numpy(), merged["entropy_decile"].to_numpy())

    lambda_priv_points = [] if args.skip_mopae else [float(x) for x in args.lambda_priv_points.split(",")]
    op_results = {}
    for lp in lambda_priv_points:
        print(f"[{args.dataset}] plain Mo-PAE (lambda4=0) at lambda_priv={lp}...", flush=True)
        bundle = train_mopae_plain(windows, total_locations, total_users, lp, device,
                                    epochs=args.max_epochs // 3 or 1)
        post_df = train_attacker_on_frozen_encoder_hardened(
            bundle.encoder, windows, total_users, device, seeds=seeds,
            max_epochs=args.max_epochs, patience=args.patience)
        colname = f"post_top1_lp{lp}"
        merged = merged.merge(post_df[["user_id", "post_top1"]].rename(columns={"post_top1": colname}),
                               on="user_id", how="left")
        merged[f"abs_reduction_lp{lp}"] = merged["raw_top1"] - merged[colname]
        merged[f"rel_reduction_lp{lp}"] = np.where(
            merged["raw_top1"] > 0, merged[f"abs_reduction_lp{lp}"] / merged["raw_top1"], np.nan)

        abs_ci = bootstrap_decile_ci(merged[f"abs_reduction_lp{lp}"].to_numpy(), merged["entropy_decile"].to_numpy())
        rel_vals = merged[f"rel_reduction_lp{lp}"].to_numpy()
        rel_deciles = merged["entropy_decile"].to_numpy()
        rel_mask = ~np.isnan(rel_vals)
        rel_ci = bootstrap_decile_ci(rel_vals[rel_mask], rel_deciles[rel_mask])

        op_results[str(lp)] = dict(
            mean_post_top1=float(merged[colname].mean()),
            mean_abs_reduction=float(merged[f"abs_reduction_lp{lp}"].mean()),
            mean_rel_reduction=float(np.nanmean(rel_vals)),
            decile_abs_reduction_ci=abs_ci,
            decile_rel_reduction_ci=rel_ci,
        )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "per_user_risk.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    deciles = sorted(decile_ci)
    means = [decile_ci[d]["mean"] for d in deciles]
    los = [decile_ci[d]["mean"] - decile_ci[d]["ci_lo"] for d in deciles]
    his = [decile_ci[d]["ci_hi"] - decile_ci[d]["mean"] for d in deciles]
    ax.errorbar(deciles, means, yerr=[los, his], marker="o", capsize=3)
    ax.set_xlabel("entropy decile (0=lowest S_unc)")
    ax.set_ylabel("mean raw top-1 re-id risk (95% bootstrap CI)")
    ax.set_title(f"{args.dataset}: raw re-id risk by entropy decile")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_by_entropy_decile.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for lp in lambda_priv_points:
        abs_ci = op_results[str(lp)]["decile_abs_reduction_ci"]
        ds = sorted(abs_ci)
        ax.plot(ds, [abs_ci[d]["mean"] for d in ds], marker="o", label=f"lambda_priv={lp}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("entropy decile (0=lowest S_unc)")
    ax.set_ylabel("absolute risk reduction (raw - post)")
    ax.set_title(f"{args.dataset}: plain Mo-PAE risk reduction by decile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "risk_reduction_by_decile.png", dpi=120)
    plt.close(fig)

    summary = dict(
        dataset=args.dataset, total_users=total_users, total_locations=total_locations,
        respect_traj_boundary=args.respect_traj_boundary,
        n_users_with_features=len(feats), n_users_in_risk_analysis=len(merged),
        sl=args.sl, coverage=coverage,
        n_train_windows=len(windows["train"]["user"]), n_test_windows=len(windows["test"]["user"]),
        attacker_winner_log=winner_log,
        raw_risk_summary=dict(mean_top1=float(merged["raw_top1"].mean()), mean_top5=float(merged["raw_top5"].mean())),
        ols_risk_vs_entropy_rog_nwindows_ncells=ols,
        decile_risk_ci=decile_ci,
        lambda_priv_operating_points=op_results,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
