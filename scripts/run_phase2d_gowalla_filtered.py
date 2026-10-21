#!/usr/bin/env python
"""Checkpoint 02d: Gowalla coverage fix. Rebuilds a Gowalla metro subset at
SL=5 (per the PI, since SL=10's test-window coverage was unusably low --
Checkpoint 3d found only 18.3% of Gowalla-NYC users passed the >=10-test-
window filter at SL=10), applies the per-user minimum filter, and reruns
the Checkpoint 2c-style disparate-risk analysis (raw vs. plain-Mo-PAE-
protected risk by entropy decile) using the PI's new primary standard:
the hardened external attacker, strongest per user across MLP/GRU/GBT
(puffin.attacks.reid's *_strongest_per_user functions), not the retired
adaptive attacker.
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
from puffin.attacks.windowing import build_sl_windows, count_sl_coverage
from puffin.attacks.reid import train_raw_attacker_strongest_per_user, train_attacker_on_frozen_encoder_strongest_per_user
from puffin.attacks.analysis import ols_with_pvalues, bootstrap_decile_ci
from puffin.models.mopae_torch import build_bundle, fixed_train_step
from puffin.metrics.privacy import privacy_gain


def train_mopae_plain(windows, total_locations, total_users, lambda_priv, device, epochs,
                       batch_size=512, lr=3e-4, seed=0):
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
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                              coord_target[idx], lambda4=0.0, lambdas=lambdas)
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sl", type=int, default=5)
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--mopae-epochs", type=int, default=20)
    ap.add_argument("--lambda-priv", type=float, default=0.3)
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--min-test-windows", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    tag = f"{args.dataset} SL={args.sl}"

    df = pd.read_parquet(args.parquet)
    # Gowalla's user_id space is sparse (gowalla_nyc's max id is 7202 for
    # only 1517 actual users) -- total_users=max(id) blew up GBT's
    # multiclass fit to a ~7202-way problem instead of the real ~1517-way
    # one, making HistGradientBoostingClassifier pathologically slow (one
    # seed's fit ran 14+ hours without finishing). Remap to a dense
    # 1..n_unique range before anything downstream touches user_id.
    unique_ids = np.sort(df["user_id"].unique())
    remap = {old: new for new, old in enumerate(unique_ids, start=1)}
    df = df.copy()
    df["user_id"] = df["user_id"].map(remap)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)

    coverage = count_sl_coverage(df, args.sl, respect_traj_boundary=False)
    print(f"[{tag}] coverage: {json.dumps(coverage)}", flush=True)

    print(f"[{tag}] building windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=False)

    print(f"[{tag}] raw attacker (strongest per user, MLP+GRU+GBT)...", flush=True)
    raw_df, raw_win_log = train_raw_attacker_strongest_per_user(windows, total_locations, total_users, device,
                                                                 seeds=seeds, max_epochs=args.attack_epochs,
                                                                 patience=6)

    print(f"[{tag}] training plain Mo-PAE (lambda4=0)...", flush=True)
    bundle = train_mopae_plain(windows, total_locations, total_users, args.lambda_priv, device,
                                epochs=args.mopae_epochs)

    print(f"[{tag}] post attacker (strongest per user, MLP+GBT on frozen encoder)...", flush=True)
    post_df, post_win_log = train_attacker_on_frozen_encoder_strongest_per_user(
        bundle.encoder, windows, total_users, device, seeds=seeds, max_epochs=args.attack_epochs, patience=6)

    merged = feats.merge(raw_df[["user_id", "n_windows", "raw_top1"]], on="user_id", how="inner")
    merged = merged.merge(post_df[["user_id", "post_top1"]], on="user_id", how="inner")
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")

    n_before = len(merged)
    excluded = merged[merged["n_windows"] < args.min_test_windows]
    filtered = merged[merged["n_windows"] >= args.min_test_windows].copy()
    n_excluded = n_before - len(filtered)
    n_total_dataset_users = df["user_id"].nunique()
    print(f"[{tag}] per-user minimum filter: {len(filtered)}/{n_before} users with risk data pass "
          f"(>={args.min_test_windows} test windows); {n_excluded} excluded. "
          f"{len(filtered)}/{n_total_dataset_users} of ALL dataset users pass.", flush=True)

    filtered["gain"] = privacy_gain(filtered["raw_top1"].to_numpy(), filtered["post_top1"].to_numpy())
    deciles = filtered["entropy_decile"].to_numpy()
    raw_ci = bootstrap_decile_ci(filtered["raw_top1"].to_numpy(), deciles)
    post_ci = bootstrap_decile_ci(filtered["post_top1"].to_numpy(), deciles)
    gain_ci = bootstrap_decile_ci(filtered["gain"].to_numpy(), deciles)

    ols = ols_with_pvalues(filtered["post_top1"].to_numpy(),
                            dict(s_unc=filtered["s_unc"].to_numpy(), rog_km=filtered["rog_km"].to_numpy(),
                                 n_windows=filtered["n_windows"].to_numpy()))

    result = dict(
        dataset=args.dataset, sl=args.sl, coverage=coverage,
        n_total_dataset_users=n_total_dataset_users, n_users_with_risk_data=n_before,
        n_users_after_min_filter=len(filtered), n_users_excluded_min_windows=n_excluded,
        pct_of_all_users_passing=round(100 * len(filtered) / n_total_dataset_users, 1),
        raw_decile_ci=raw_ci, post_decile_ci=post_ci, gain_decile_ci=gain_ci,
        ols_post_top1_on_s_unc_rog_nwindows=ols,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_dir / "per_user_filtered.csv", index=False)
    excluded.to_csv(out_dir / "excluded_users.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("raw_decile_ci", "post_decile_ci", "gain_decile_ci")}, indent=2))


if __name__ == "__main__":
    main()
