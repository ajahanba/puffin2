"""Unified trajectory schema shared by every dataset loader.

user_id   : dense int, 1-indexed, unique per person
traj_id   : dense int, 1-indexed *within* user_id (not globally unique -- always
            group by (user_id, traj_id) together)
ts        : pandas Timestamp when the dataset has one; otherwise a monotonic
            integer surrogate (see each loader's `has_real_timestamp` flag --
            this matters for temporal splits, see puffin/data/splits.py)
lat, lon  : float, nullable (null for grid-only datasets like YJMob100K)
cell_id   : dense int, 1-indexed, the tessellation cell (see tessellation.py)
weekday   : int 0-6, nullable
hour      : int 0-23, nullable
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"]


def validate_schema(df: pd.DataFrame, allow_null_latlon: bool = False) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df["user_id"].isna().any() or df["traj_id"].isna().any() or df["cell_id"].isna().any():
        raise ValueError("user_id, traj_id, cell_id must never be null")
    if not allow_null_latlon and (df["lat"].isna().any() or df["lon"].isna().any()):
        raise ValueError("lat/lon are null but allow_null_latlon=False")
    if (df["user_id"] < 1).any() or (df["traj_id"] < 1).any() or (df["cell_id"] < 1).any():
        raise ValueError("user_id, traj_id, cell_id must be 1-indexed (>=1)")
