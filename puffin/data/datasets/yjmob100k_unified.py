"""YJMob100K (Zenodo 10.5281/zenodo.10836269, Yabe et al. 2024) -> unified schema.

Raw format confirmed directly from the downloaded file (header present):
uid, d (day 0-74), t (30-minute timeslot 0-47), x, y (500m grid cell, city
undisclosed). dataset1.csv has 111,535,176 rows, 100,000 unique uids
(80,000 "regular" + 20,000 "task" users whose later days were masked in the
original prediction challenge -- see data/raw/yjmob100k/SOURCE.md). No real
lat/lon (HAS_REAL_TIMESTAMP = True via d/t, but no H3 sensitivity check is
possible here, same limitation as GeoLife's sample, for the opposite reason:
no coordinates at all rather than already-gridded ones).

`ts` = d*48 + t, a genuinely monotonic, evenly-spaced relative time (NOT a
wall-clock timestamp, since the start date/city is undisclosed by design).
`hour` = t // 2. `weekday` is left null -- there is no way to anchor day 0 to
a real calendar date. `cell_id` is the dataset's own (x, y) grid cell, dense
1-indexed, same "pass through the dataset's own grid" treatment as GeoLife's
`loc` column -- no re-tessellation, since x/y already IS the grid.
`traj_id` = day index (d), dense per user (handles the rare case where a
user's observed days aren't contiguous from 0).

This file is large (111M rows, ~2.1GB uncompressed): `nrows` lets you load a
prefix for quick iteration; pass None for the full file.
"""
from __future__ import annotations

import pandas as pd

HAS_REAL_TIMESTAMP = True


def load_yjmob100k_unified(path: str, nrows: int | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path, nrows=nrows)

    df = pd.DataFrame()
    df["user_id"] = raw["uid"] + 1  # dense already (0-indexed uid -> 1-indexed user_id)
    df["ts"] = raw["d"] * 48 + raw["t"]
    df["lat"] = float("nan")
    df["lon"] = float("nan")
    df["weekday"] = pd.NA
    df["hour"] = raw["t"] // 2

    # Vectorized: this file can be 100M+ rows, so no Python-level per-row loops.
    xy_key = raw["x"].astype("int64") * (int(raw["y"].max()) + 1) + raw["y"].astype("int64")
    codes, _ = pd.factorize(xy_key, sort=True)
    df["cell_id"] = codes + 1

    # .to_numpy(), not a bare Series: a Series here would carry traj_key's own
    # fresh 0..n-1 index, which silently misaligns against df's index if df
    # ever stops being a contiguous 0..n-1 range (see _checkin_utils.py for
    # the bug this caused in gowalla_unified.py). Currently safe either way
    # since nothing filters `raw` before this point, but kept robust anyway.
    traj_key = pd.DataFrame({"user_id": df["user_id"].values, "d": raw["d"].values})
    df["traj_id"] = traj_key.groupby("user_id")["d"].rank(method="dense").astype(int).to_numpy()

    return df[["user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"]]
