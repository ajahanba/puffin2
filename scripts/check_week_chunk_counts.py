#!/usr/bin/env python
"""Reports train/test chunk counts under week-grouping (vs the old
trajectory-grouping) for Foursquare and Gowalla-SF-Bay, WITHOUT training
any GAN -- per the PI's instruction to report new chunk counts before
committing to full 3-seed runs."""
import sys

import pandas as pd

sys.path.insert(0, ".")
from puffin.data.tessellation import assign_grid_cells
from puffin.baselines.trajgan import _trajectory_chunks

DATASETS = {
    "foursquare": "data/processed/foursquare/unified.parquet",
    "gowalla_sf_bay": "data/processed/gowalla_sf_bay/unified.parquet",
}

for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    df = assign_grid_cells(df, resolution_deg=0.01)
    n_users = df["user_id"].nunique()
    print(f"\n=== {name} (n_users={n_users}, n_points={len(df)}) ===", flush=True)
    for group_by, train_min_traj_len in [("trajectory", 11), ("week", 2)]:
        train_chunks = _trajectory_chunks(df, "train", max_traj_len=150,
                                           min_traj_len=train_min_traj_len, group_by=group_by)
        test_chunks = _trajectory_chunks(df, "test", max_traj_len=150,
                                          min_traj_len=1, group_by=group_by)
        lens = [g["length"] for g in train_chunks]
        mean_len = sum(lens) / len(lens) if lens else 0.0
        print(f"  group_by={group_by:10s} train_min_traj_len={train_min_traj_len}: "
              f"n_train_chunks={len(train_chunks)}, n_test_chunks={len(test_chunks)}, "
              f"mean_train_chunk_len={mean_len:.1f}", flush=True)
