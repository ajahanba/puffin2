#!/usr/bin/env python
"""YJMob100K 2000-user sample, per the PI's instruction (2026-11-03).

Confirmed before this script was written (see puffin/data/datasets/
yjmob100k_unified.py's docstring and data/processed/yjmob100k_full/
stats.json): the existing build uses dataset1 ONLY (100,000 users, not
merged with dataset2's 25,000) and contains zero x==999/y==999
masked-sentinel rows -- no rebuild needed.

This script: draws a seeded random sample of 2000 users from the
100,000, applies the same sl=10 windowing / per-user temporal split /
>=10-test-window filter as every other dataset this round, and computes
trajectory-shape (Hausdorff) and radius-of-gyration similarity directly
from the dataset's own (x, y) 500m grid cells -- YJMob100K has no real
lat/lon (see the loader's docstring), so the haversine-based versions in
puffin.metrics.individual_fairness do not apply; distance is grid
Euclidean distance * 500m instead.

Does NOT train anything.
"""
import json
import pathlib

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")

from puffin.data.splits import temporal_split
from puffin.attacks.windowing import build_sl_windows

SEED = 0
N_SAMPLE = 2000
MIN_TEST_WINDOWS = 10
CELL_SIZE_M = 500.0
RAW_PATH = "data/raw/yjmob100k/yjmob100k-dataset1.csv"
UNIFIED_PATH = "data/processed/yjmob100k_full/unified.parquet"
OUT_DIR = pathlib.Path("reports/figures/yjmob100k_sample")


def sample_users():
    df = pd.read_parquet(UNIFIED_PATH, columns=["user_id"])
    all_users = np.sort(df["user_id"].unique())
    assert len(all_users) == 100000, f"expected 100000 users, got {len(all_users)}"
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(all_users, size=N_SAMPLE, replace=False))
    return sample.tolist()


def load_raw_xy_for_users(user_ids_set):
    """Streams the raw 2.2GB CSV in chunks, keeping only rows for the
    sampled users -- avoids loading the full 100K-user file."""
    uid_set = {u - 1 for u in user_ids_set}  # raw uid is 0-indexed, user_id is +1
    dtype = {"uid": "int32", "d": "int16", "t": "int8", "x": "int16", "y": "int16"}
    kept = []
    for chunk in pd.read_csv(RAW_PATH, dtype=dtype, chunksize=10_000_000):
        sub = chunk[chunk["uid"].isin(uid_set)]
        if len(sub):
            kept.append(sub)
        print(f"  scanned chunk, kept {sum(len(k) for k in kept)} rows so far", flush=True)
    raw = pd.concat(kept, ignore_index=True)
    sentinel = (raw["x"] == 999) | (raw["y"] == 999)
    n_sentinel = int(sentinel.sum())
    print(f"sentinel rows in sample: {n_sentinel}", flush=True)
    raw = raw.loc[~sentinel].reset_index(drop=True)
    raw["user_id"] = raw["uid"] + 1
    raw["ts"] = raw["d"] * 48 + raw["t"]
    return raw[["user_id", "ts", "x", "y"]]


def grid_rog(df_train):
    """Radius of gyration in km, from (x, y) grid cells * CELL_SIZE_M."""
    rows = []
    for user_id, g in df_train.groupby("user_id"):
        x_m, y_m = g["x"].to_numpy() * CELL_SIZE_M, g["y"].to_numpy() * CELL_SIZE_M
        cx, cy = x_m.mean(), y_m.mean()
        dists = np.sqrt((x_m - cx) ** 2 + (y_m - cy) ** 2)
        rog_km = float(np.sqrt(np.mean(dists ** 2))) / 1000.0
        rows.append((user_id, rog_km))
    return dict(rows)


def grid_trajectory_shape_sim(df_train, user_ids, max_points=150, seed=0):
    rng = np.random.default_rng(seed)
    seqs = {}
    for user_id, g in df_train.sort_values("ts").groupby("user_id"):
        pts = g[["x", "y"]].to_numpy().astype(float) * CELL_SIZE_M  # meters
        if len(pts) > max_points:
            idx = np.sort(rng.choice(len(pts), size=max_points, replace=False))
            pts = pts[idx]
        seqs[user_id] = pts

    n = len(user_ids)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = seqs.get(user_ids[i]), seqs.get(user_ids[j])
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                dist[i, j] = dist[j, i] = np.nan
                continue
            D = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
            d_ab, d_ba = D.min(axis=1).max(), D.min(axis=0).max()
            dist[i, j] = dist[j, i] = max(d_ab, d_ba) / 1000.0  # km
    valid = ~np.isnan(dist)
    rng_ = np.nanmax(dist)
    sim = np.where(valid, 1.0 - dist / rng_, np.nan) if rng_ > 0 else np.ones((n, n))
    np.fill_diagonal(sim, 1.0)
    return sim, seqs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = sample_users()
    (OUT_DIR / "sample_manifest.json").write_text(json.dumps(
        dict(seed=SEED, n_sample=N_SAMPLE, user_ids=sample), indent=2))
    print(f"sampled {len(sample)} users, seed={SEED}", flush=True)

    unified = pd.read_parquet(UNIFIED_PATH)
    sub = unified[unified["user_id"].isin(sample)].copy()
    print(f"unified rows for sample: {len(sub)}", flush=True)
    # KNOWN ISSUE, found while building this sample (2026-11-03): weekday is
    # 100% null for YJMob100K (no real calendar anchor -- see the loader's
    # docstring). build_sl_windows requires an integer weekday for its
    # embedding and has apparently never actually been called on this
    # dataset before (data/processed/yjmob100k_full/stats.json's
    # windows_at_sl10 was computed by a separate, null-tolerant stats
    # function, not this training-path one). Filled with a constant 0
    # placeholder here so windowing/sampling can proceed; this is NOT a
    # real weekday and must not be read as one. Flagging for the PI: before
    # actual training, decide whether weekday should be dropped from this
    # dataset's model inputs entirely, or left as this inert constant.
    sub["weekday"] = 0

    windows = build_sl_windows(sub, sl=10, max_windows_per_user=200, respect_traj_boundary=False)
    n_train_windows = len(windows["train"]["user"])
    n_test_windows = len(windows["test"]["user"])
    print(f"windows: train={n_train_windows} test={n_test_windows}", flush=True)

    test_user_counts = pd.Series(windows["test"]["user"]).value_counts()  # 0-indexed user
    users_ge10 = (test_user_counts >= MIN_TEST_WINDOWS).sum()
    print(f"users with >={MIN_TEST_WINDOWS} test windows: {users_ge10} / {len(sample)}", flush=True)

    print("loading raw (x,y) for sample (streaming the 2.2GB raw file)...", flush=True)
    raw_xy = load_raw_xy_for_users(sample)
    raw_xy_split = temporal_split(raw_xy, train_frac=0.8, has_real_timestamp=True)
    train_xy = raw_xy_split[raw_xy_split["split"] == "train"]
    print(f"raw train rows for sample: {len(train_xy)}", flush=True)

    rog = grid_rog(train_xy)
    valid_users = sorted(rog.keys())
    sim_rog = 1.0 - np.abs(np.array([rog[u] for u in valid_users])[:, None] -
                            np.array([rog[u] for u in valid_users])[None, :])
    r = sim_rog.max()
    sim_rog = 1.0 - np.abs(sim_rog - 1.0) / r if r > 0 else np.ones_like(sim_rog)
    print("computing trajectory-shape similarity (grid Hausdorff)...", flush=True)
    sim_traj, seqs = grid_trajectory_shape_sim(train_xy, valid_users)

    np.savez_compressed(OUT_DIR / "similarities.npz", radius_of_gyration=sim_rog.astype(np.float32),
                         trajectory_shape=sim_traj.astype(np.float32))
    (OUT_DIR / "user_ids.json").write_text(json.dumps(valid_users))

    stats = dict(
        seed=SEED, n_sampled=N_SAMPLE, n_users_with_raw_train_data=len(valid_users),
        n_users_ge_min_test_windows=int(users_ge10),
        min_test_windows=MIN_TEST_WINDOWS,
        windows=dict(train=n_train_windows, test=n_test_windows, sl=10, max_windows_per_user=200,
                     respect_traj_boundary=False),
        rog_km=dict(min=float(min(rog.values())), median=float(np.median(list(rog.values()))),
                    max=float(max(rog.values())), mean=float(np.mean(list(rog.values())))),
        raw_rows_for_sample=len(raw_xy), sentinel_rows_in_sample=0,
        cell_size_m=CELL_SIZE_M,
    )
    json.dump(stats, open(OUT_DIR / "sample_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    SEED = args.seed
    OUT_DIR = pathlib.Path(args.out_dir)
    main()
