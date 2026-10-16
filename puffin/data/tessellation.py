"""Two tessellations, per Phase 1 task 2: a 0.01-degree grid (continuity with
Mo-PAE/PUFFIN, which used a fixed-size grid) and H3 resolution 8 as a
sensitivity check. Both produce a dense, 1-indexed `cell_id` column so
downstream code never has to know which tessellation was used.
"""
from __future__ import annotations

import pandas as pd


def assign_grid_cells(df: pd.DataFrame, resolution_deg: float = 0.01) -> pd.DataFrame:
    """Floor (lat, lon) to a resolution_deg x resolution_deg cell, then remap
    to a dense 1-indexed cell_id. Requires real lat/lon (not usable for
    grid-only datasets like YJMob100K, which already ship a cell id)."""
    df = df.copy()
    lat_idx = (df["lat"] / resolution_deg).apply(lambda x: int(x // 1))
    lon_idx = (df["lon"] / resolution_deg).apply(lambda x: int(x // 1))
    key = list(zip(lat_idx, lon_idx))
    uniq = {k: i + 1 for i, k in enumerate(sorted(set(key)))}
    df["cell_id"] = [uniq[k] for k in key]
    return df


def assign_h3_cells(df: pd.DataFrame, resolution: int = 8) -> pd.DataFrame:
    """Sensitivity-check tessellation. Adds `cell_id_h3` (dense 1-indexed),
    leaving `cell_id` (whatever the primary tessellation produced) untouched,
    so both can be compared side by side.

    `h3.latlng_to_cell` is called once per UNIQUE (lat, lon) pair, not once
    per row -- the Python h3 bindings aren't vectorized, and Gowalla's first
    full run took ~17 minutes at one call per row (3.67M rows) where the
    actual number of distinct coordinates is far smaller. Results are mapped
    back to every row afterward."""
    import h3  # optional dependency, only needed for the sensitivity check

    df = df.copy()
    uniq_coords = df[["lat", "lon"]].drop_duplicates()
    h3_by_coord = {
        (lat, lon): h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(uniq_coords["lat"], uniq_coords["lon"])
    }
    h3_idx = [h3_by_coord[(lat, lon)] for lat, lon in zip(df["lat"], df["lon"])]
    uniq_cells = {h: i + 1 for i, h in enumerate(sorted(set(h3_idx)))}
    df["cell_id_h3"] = [uniq_cells[h] for h in h3_idx]
    return df
