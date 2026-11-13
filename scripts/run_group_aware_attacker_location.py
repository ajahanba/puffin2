#!/usr/bin/env python
"""Group-aware attacker for location-PUFFIN (lambda4=0.3) vs Mo-PAE, same
procedure as scripts/run_item7_v2.py: fixed decile membership (entropy,
computed once over all GeoLife users), a freshly-trained frozen encoder
at seed 0, and one group-restricted external attacker per decile (5
attacker seeds). Mo-PAE's decile group-attacker CSVs already exist
(geolife_mopae_matched_lambda3_1.2_decile*_group_attacker/) and are
reused unmodified -- only location-PUFFIN's group attacker is new here.
"""
import functools
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_attacker_on_frozen_encoder_strongest_per_user
from puffin.fairness.losses import build_entropy_table, build_histogram_table, ed_soft_location
from scripts.run_phase4d_approved_batch import train_mechanism, eval_utility

DEVICE = torch.device("cuda:1")
PARQUET = "data/processed/geolife/unified.parquet"
OUT_DIR = pathlib.Path("reports/figures/paper")
ATTACKER_SEEDS = (0, 1, 2, 3, 4)
ENCODER_SEED = 0
LAMBDA4 = 0.3
LAMBDA_PRIV = 0.3
FAIRNESS_RESCALE = 16663410.0
GROUP_LABEL = "puffin_location_lambda4_0.3"


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

    train_df = ifm.train_only(df)
    hist_by_user = ifm.per_user_cell_histogram(train_df, total_locations)
    fairness_table = build_histogram_table(hist_by_user, total_users, total_locations, device=DEVICE)
    fairness_fn_override = functools.partial(ed_soft_location, rescale=FAIRNESS_RESCALE)

    print("building windows...", flush=True)
    windows = build_sl_windows(df, sl=10, max_windows_per_user=200, respect_traj_boundary=False)

    feats_decile = feats.copy()
    feats_decile["entropy_decile"] = pd.qcut(feats_decile["s_unc"], 10, labels=False, duplicates="drop")
    decile_by_user_0idx = {u - 1: d for u, d in zip(feats_decile["user_id"], feats_decile["entropy_decile"])}

    print(f"\n=== {GROUP_LABEL}: training encoder seed={ENCODER_SEED} ===", flush=True)
    t0 = time.time()
    bundle, actual_epochs = train_mechanism(
        windows, total_locations, total_users, fairness_table, "ed_soft", LAMBDA4,
        DEVICE, max_epochs=80, lambda_priv=LAMBDA_PRIV, seed=ENCODER_SEED,
        patience=8, val_frac=0.1, min_epochs=20, fairness_fn_override=fairness_fn_override)
    utility = eval_utility(bundle, windows, DEVICE)
    print(f"{GROUP_LABEL}: stopped after {actual_epochs} epochs, utility={utility:.4f}, "
          f"trained in {time.time()-t0:.1f}s", flush=True)

    for d in sorted(set(decile_by_user_0idx.values())):
        decile_users_0idx = {u for u, dd in decile_by_user_0idx.items() if dd == d}
        dec_windows = filter_windows_to_users(windows, decile_users_0idx)
        if len(dec_windows["test"]["user"]) == 0:
            continue
        post_df, _ = train_attacker_on_frozen_encoder_strongest_per_user(
            bundle.encoder, dec_windows, total_users, DEVICE, seeds=ATTACKER_SEEDS,
            max_epochs=25, patience=6, attacker_device=torch.device("cpu"))
        out_name = f"geolife_{GROUP_LABEL}_decile{d}_group_attacker"
        out_dir = OUT_DIR / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        post_df.to_csv(out_dir / "per_user_04d.csv", index=False)
        print(f"  decile {d}: saved {out_dir}/per_user_04d.csv, n_users={len(decile_users_0idx)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
