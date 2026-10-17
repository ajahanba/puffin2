#!/usr/bin/env python
"""Phase 2 (Checkpoint 2): is re-identification risk unequal across users,
and does Mo-PAE (fixed) protect the people who need it most -- or least?

For one dataset (its data/processed/<name>/unified.parquet from Phase 1):
  1. Per-user mobility features (puffin.features.mobility): S_rand, S_unc,
     S_real, radius of gyration, regularity.
  2. A fresh "raw" re-id attacker (puffin.attacks.reid.train_raw_attacker) on
     unprotected SL=10 windows -> per-user risk with no mechanism at all.
  3. A risk ~ entropy + rog + n_records regression (plain OLS via
     numpy.linalg.lstsq -- no new dependency).
  4. Mo-PAE (fixed mode, puffin.models.mopae_torch) trained at a few
     lambda4 (fairness weight) operating points, each followed by a FRESH
     attacker trained on that run's frozen encoder output
     (train_attacker_on_frozen_encoder) -- never the in-model adversary.
  5. Per-user risk change (after vs. before) by entropy decile: who gains,
     who loses, at each operating point.

Writes a JSON blob of everything (for the report) and two PNGs (risk vs.
entropy decile; risk vs. regularity) to the given --out-dir.
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
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker, train_attacker_on_frozen_encoder
from puffin.models.mopae_torch import build_bundle, fixed_train_step
from puffin.fairness.losses import ed_soft, build_entropy_table


def fit_ols(y: np.ndarray, X_cols: dict) -> dict:
    names = list(X_cols.keys())
    X = np.column_stack([np.ones(len(y))] + [X_cols[n] for n in names])
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(intercept=float(coef[0]), coefficients={n: float(c) for n, c in zip(names, coef[1:])}, r2=r2)


def train_mopae_fixed(df: pd.DataFrame, windows: dict, total_locations: int, total_users: int,
                       entropy_table: torch.Tensor, lambda4: float, device, epochs: int,
                       batch_size: int = 512, lr: float = 3e-4):
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
    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)  # real next-location target
    # No real (lat, lon) reconstruction target is wired up for this Phase 2 check (that
    # would need per-window coordinates threaded through windowing.py); reconstruction
    # loss is computed against zeros, i.e. lambda_recon's term is present but uninformative.
    # This does not affect what Phase 2 is measuring (identification risk vs. a REAL
    # next-location utility objective + a real privacy adversary); flagged here rather
    # than silently assumed harmless.
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx],
                              y_next_idx[idx], user_idx[idx], coord_target[idx],
                              fairness_fn=ed_soft, entropy_table=entropy_table, lambda4=lambda4)
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--attacker-epochs", type=int, default=5)
    ap.add_argument("--mopae-epochs", type=int, default=5)
    ap.add_argument("--lambda4-points", type=str, default="0.0,0.3,1.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())

    print(f"[{args.dataset}] computing mobility features for {total_users} users...", flush=True)
    feats = compute_mobility_features(df)

    print(f"[{args.dataset}] building SL={args.sl} windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user, seed=args.seed)
    print(f"[{args.dataset}] train windows={len(windows['train']['user'])}, test windows={len(windows['test']['user'])}", flush=True)

    print(f"[{args.dataset}] training raw (unprotected) attacker...", flush=True)
    raw_encoder, raw_head, raw_risk = train_raw_attacker(
        windows, total_locations, total_users, device, epochs=args.attacker_epochs)
    raw_risk_df = pd.DataFrame(raw_risk).rename(columns={"top1": "raw_top1", "top5": "raw_top5"})

    merged = feats.merge(raw_risk_df, on="user_id", how="inner")
    print(f"[{args.dataset}] {len(merged)} users have both features and >=1 test window "
          f"(of {total_users} total, {len(feats)} with features)", flush=True)

    ols = fit_ols(merged["raw_top1"].to_numpy(), dict(
        s_unc=merged["s_unc"].to_numpy(), rog_km=merged["rog_km"].to_numpy(),
        n_records=merged["n_records"].to_numpy()))

    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")
    decile_risk = merged.groupby("entropy_decile")["raw_top1"].mean().to_dict()

    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

    operating_points = [float(x) for x in args.lambda4_points.split(",")]
    op_results = {}
    for lambda4 in operating_points:
        print(f"[{args.dataset}] training Mo-PAE (fixed) at lambda4={lambda4}...", flush=True)
        bundle = train_mopae_fixed(df, windows, total_locations, total_users, entropy_table,
                                    lambda4, device, epochs=args.mopae_epochs)
        attacker_head, post_risk = train_attacker_on_frozen_encoder(
            bundle.encoder, windows, total_users, device, epochs=args.attacker_epochs)
        post_df = pd.DataFrame(post_risk).rename(columns={"top1": f"post_top1_l{lambda4}", "top5": f"post_top5_l{lambda4}"})
        merged = merged.merge(post_df[["user_id", f"post_top1_l{lambda4}"]], on="user_id", how="left")
        merged[f"delta_l{lambda4}"] = merged[f"post_top1_l{lambda4}"] - merged["raw_top1"]
        op_results[str(lambda4)] = dict(
            mean_post_top1=float(merged[f"post_top1_l{lambda4}"].mean()),
            mean_delta=float(merged[f"delta_l{lambda4}"].mean()),
            frac_helped=float((merged[f"delta_l{lambda4}"] < -1e-9).mean()),
            frac_harmed=float((merged[f"delta_l{lambda4}"] > 1e-9).mean()),
            frac_unchanged=float((merged[f"delta_l{lambda4}"].abs() <= 1e-9).mean()),
            decile_mean_delta={int(k): float(v) for k, v in
                                merged.groupby("entropy_decile")[f"delta_l{lambda4}"].mean().to_dict().items()},
        )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "per_user_risk.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    deciles = sorted(decile_risk)
    ax.plot(deciles, [decile_risk[d] for d in deciles], marker="o")
    ax.set_xlabel("entropy decile (0=lowest S_unc)")
    ax.set_ylabel("mean raw top-1 re-id risk")
    ax.set_title(f"{args.dataset}: raw re-id risk by entropy decile")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_by_entropy_decile.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["regularity_top2"], merged["raw_top1"], alpha=0.5, s=10)
    ax.set_xlabel("regularity (share of visits at top-2 locations)")
    ax.set_ylabel("raw top-1 re-id risk")
    ax.set_title(f"{args.dataset}: raw re-id risk vs. regularity")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_vs_regularity.png", dpi=120)
    plt.close(fig)

    summary = dict(
        dataset=args.dataset, total_users=total_users, total_locations=total_locations,
        n_users_with_features=len(feats), n_users_in_risk_analysis=len(merged),
        sl=args.sl, max_windows_per_user=args.max_windows_per_user,
        n_train_windows=len(windows["train"]["user"]), n_test_windows=len(windows["test"]["user"]),
        raw_risk_summary=dict(mean_top1=float(merged["raw_top1"].mean()),
                              mean_top5=float(merged["raw_top5"].mean())),
        ols_risk_vs_entropy_rog_nrecords=ols,
        decile_risk=({int(k): float(v) for k, v in decile_risk.items()}),
        operating_points=op_results,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
