"""YJMob100K (Zenodo 10.5281/zenodo.10836269, Yabe et al. 2024) -> unified schema.

Raw format confirmed directly from the downloaded file (header present):
uid, d (day 0-74), t (30-minute timeslot 0-47), x, y (500m grid cell, city
undisclosed). dataset1.csv has 111,535,175 data rows, 100,000 unique uids.

Checked directly: there is no 80K/20K split visible in this file by any
mechanism checked. No x==999/y==999 sentinel rows exist (0 found).
Per-user day coverage is unimodal, not bimodal: 97,836 of 100,000 users
reach day 74 (the max), the remaining 2,164 have less (min 18 days),
with no bimodal split by max-day or day-count that would separate
"regular" from "task" users. The Zenodo page's text describes dataset1
as "80,000 people, business-as-usual" but the file plainly contains
100,000 unique uids -- this is an unresolved discrepancy between the
page's prose and the actual file, not something this loader can
reconcile; treat dataset1 as one 100,000-user population. dataset2 is
not used by this loader.

No real lat/lon (HAS_REAL_TIMESTAMP = True via d/t, but no H3 sensitivity
check is possible here, same limitation as GeoLife's sample, for the
opposite reason: no coordinates at all rather than already-gridded ones).

`ts` = d*48 + t, a genuinely monotonic, evenly-spaced relative time (NOT a
wall-clock timestamp, since the start date/city is undisclosed by design).
`hour` = t // 2. `weekday` = day_index % 7: there is no way to anchor day
0 to a real calendar date, so this is NOT a real day-of-week -- it is a
7-day-periodic proxy so the model can still learn weekly structure if
any exists, at an arbitrary, unknown phase offset; do not read this
column as calendar truth. `cell_id` is the dataset's own (x, y) grid cell, dense
1-indexed, same "pass through the dataset's own grid" treatment as GeoLife's
`loc` column -- no re-tessellation, since x/y already IS the grid.
`traj_id` = day index (d), dense per user (handles the rare case where a
user's observed days aren't contiguous from 0).

This file is large (111.5M rows). `nrows` loads a prefix for quick
iteration. For the full file, reads in `chunksize`-row chunks with compact
dtypes (int32/int16/int8) rather than one single read, then concatenates --
on a 187GB-RAM box this is not strictly required for memory, but keeps peak
RSS lower and bounded regardless of host. x==999/y==999 (the documented
masking sentinel for this dataset family, even though none were found in
*this* file) are dropped and counted, logged via `print` so a background
run's log captures it.
"""
from __future__ import annotations

import pandas as pd

HAS_REAL_TIMESTAMP = True

_DTYPE = {"uid": "int32", "d": "int16", "t": "int8", "x": "int16", "y": "int16"}


def load_yjmob100k_unified(path: str, nrows: int | None = None, chunksize: int | None = 10_000_000) -> pd.DataFrame:
    if nrows is not None or chunksize is None:
        raw = pd.read_csv(path, nrows=nrows, dtype=_DTYPE)
    else:
        chunks = list(pd.read_csv(path, dtype=_DTYPE, chunksize=chunksize))
        raw = pd.concat(chunks, ignore_index=True)
        del chunks

    sentinel = (raw["x"] == 999) | (raw["y"] == 999)
    n_sentinel = int(sentinel.sum())
    if n_sentinel:
        print(f"load_yjmob100k_unified: dropping {n_sentinel} masked-sentinel (x==999 or y==999) rows")
        raw = raw.loc[~sentinel].reset_index(drop=True)
    else:
        print("load_yjmob100k_unified: no x==999/y==999 sentinel rows found")

    df = pd.DataFrame()
    df["user_id"] = raw["uid"] + 1  # dense already (0-indexed uid -> 1-indexed user_id)
    df["ts"] = raw["d"] * 48 + raw["t"]
    df["lat"] = float("nan")
    df["lon"] = float("nan")
    df["weekday"] = raw["d"].to_numpy() % 7  # no real calendar anchor; see module docstring
    df["hour"] = raw["t"] // 2

    # Vectorized: this file can be 100M+ rows, so no Python-level per-row loops.
    xy_key = raw["x"].astype("int64") * (int(raw["y"].max()) + 1) + raw["y"].astype("int64")
    codes, _ = pd.factorize(xy_key, sort=True)
    df["cell_id"] = codes + 1

    # .to_numpy(), not a bare Series: a Series here would carry traj_key's own
    # fresh 0..n-1 index, which would silently misalign against df's index
    # if df ever stops being a contiguous 0..n-1 range. Currently safe
    # either way since nothing filters `raw` before this point, but kept
    # robust anyway.
    traj_key = pd.DataFrame({"user_id": df["user_id"].values, "d": raw["d"].values})
    df["traj_id"] = traj_key.groupby("user_id")["d"].rank(method="dense").astype(int).to_numpy()

    return df[["user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"]]
