#!/usr/bin/env python
"""Item 7 (PI instruction, 2026-11-09): GeoLife, after items 2-3. Train
one attacker per entropy-decile group (only that group's users, same
architectures/seeds as the standard attacker: MLP+GRU+GBT, 5 seeds), on
Mo-PAE at the matched lambda3 (=1.2, utility 0.721 vs. PUFFIN's 0.723 --
the closest point on the extended curve) and on PUFFIN lambda4=1.0.
Report per-decile risk vs. the standard (full-population) attacker.

No encoder checkpoint was ever saved to disk by run_phase4d_approved_batch.py
(confirmed) -- retrains ONE encoder seed (seed=0) per mechanism here
specifically to produce a frozen encoder to attack with group-restricted
attackers; the "standard attacker" comparison point reuses the EXISTING
10-seed run's own per-decile post_top1 (no retrain needed for that side).
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_attacker_on_frozen_encoder_strongest_per_user
from puffin.fairness.losses import build_entropy_table
from scripts.run_phase4d_approved_batch import train_mechanism, eval_utility

DEVICE = torch.device("cuda:1")
PARQUET = "data/processed/geolife/unified.parquet"
RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
OUT_DIR = pathlib.Path("reports/figures/paper")
ATTACKER_SEEDS = (0, 1, 2, 3, 4)
ENCODER_SEED = 0

MECHS = [
    dict(label="mopae_matched_lambda3_1.2", mechanism="plain", lambda4=0.0, lambda_priv=1.2,
         standard_run="geolife_plain_lp1.2_10seed"),
    dict(label="puffin_lambda4_1.0", mechanism="ed_soft", lambda4=1.0, lambda_priv=0.3,
         standard_run="geolife_ed_soft_lambda1.0_10seed"),
]


def filter_windows_to_users(windows, user_idx_0based_set):
    out = {}
    for split in ("train", "test"):
        w = windows[split]
        mask = np.isin(w["user"], list(user_idx_0based_set))
        out[split] = {k: v[mask] for k, v in w.items()}
    return out


def main():
    df = pd.read_parquet(PARQUET)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=DEVICE)

    print("building windows...", flush=True)
    windows = build_sl_windows(df, sl=10, max_windows_per_user=200, respect_traj_boundary=False)

    # decile assignment, same convention as per_user_04d.csv (qcut on train-split entropy)
    feats_decile = feats.copy()
    feats_decile["entropy_decile"] = pd.qcut(feats_decile["s_unc"], 10, labels=False, duplicates="drop")
    decile_by_user_1idx = dict(zip(feats_decile["user_id"], feats_decile["entropy_decile"]))
    # windows' "user" field is 0-indexed (user_id - 1)
    decile_by_user_0idx = {u - 1: d for u, d in decile_by_user_1idx.items()}

    rows = []
    for mech in MECHS:
        print(f"\n=== {mech['label']}: training encoder seed={ENCODER_SEED} ===", flush=True)
        t0 = time.time()
        bundle, actual_epochs = train_mechanism(
            windows, total_locations, total_users, entropy_table, mech["mechanism"], mech["lambda4"],
            DEVICE, max_epochs=80, lambda_priv=mech["lambda_priv"], seed=ENCODER_SEED,
            patience=8, val_frac=0.1, min_epochs=20)
        utility = eval_utility(bundle, windows, DEVICE)
        print(f"{mech['label']}: stopped after {actual_epochs} epochs, utility={utility:.4f}, "
              f"trained in {time.time()-t0:.1f}s", flush=True)

        standard_df = pd.read_csv(RETRAIN_BASE / mech["standard_run"] / "per_user_04d.csv")
        standard_df = standard_df[standard_df.n_windows >= 10]

        for d in sorted(set(decile_by_user_0idx.values())):
            decile_users_0idx = {u for u, dd in decile_by_user_0idx.items() if dd == d}
            dec_windows = filter_windows_to_users(windows, decile_users_0idx)
            n_train, n_test = len(dec_windows["train"]["user"]), len(dec_windows["test"]["user"])
            if n_test == 0:
                print(f"  decile {d}: no test windows, skipping", flush=True)
                continue
            post_df, _ = train_attacker_on_frozen_encoder_strongest_per_user(
                bundle.encoder, dec_windows, total_users, DEVICE, seeds=ATTACKER_SEEDS,
                max_epochs=25, patience=6, attacker_device=torch.device("cpu"))
            group_risk = float(post_df["post_top1"].mean())
            standard_risk = float(standard_df.loc[standard_df.entropy_decile == d, "post_top1"].mean())
            rows.append(dict(mechanism=mech["label"], decile=int(d), n_users_in_group=len(decile_users_0idx),
                              n_test_windows=n_test, group_aware_attacker_risk=group_risk,
                              standard_attacker_risk=standard_risk, delta=group_risk - standard_risk))
            print(f"  decile {d}: n_users={len(decile_users_0idx)}, group_risk={group_risk:.4f}, "
                  f"standard_risk={standard_risk:.4f}, delta={group_risk-standard_risk:+.4f}", flush=True)

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "item7_group_aware_attacker.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}, {len(out_df)} rows")


if __name__ == "__main__":
    main()
