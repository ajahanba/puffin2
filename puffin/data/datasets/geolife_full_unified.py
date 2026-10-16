"""Raw Microsoft GeoLife GPS Trajectories 1.3 (182 users, real lat/lon) ->
unified schema. NOT the small pre-gridded Mo-PAE sample
(`puffin.data.datasets.geolife_unified`, kept separately for Phase 0
legacy-mode comparisons only) -- this is the actual Microsoft download,
confirmed directly (see data/raw/geolife_full/SOURCE.md and the agent report
that fetched it): 182 user folders (`Data/000/` .. `Data/181/`), each with a
`Trajectory/` subfolder of `.plt` files (18,670 total), one file per
recorded trajectory.

`.plt` format confirmed by reading real files from 3 different users: 6
boilerplate header lines, then comma-separated rows of
lat, lon, 0 (unused), altitude (feet, unused here), date-serial (unused,
redundant with the next two columns), date string ("2008-10-23"), time
string ("02:53:04"). Real lat/lon and a real timestamp (HAS_REAL_TIMESTAMP =
True) -- this is what makes the H3 sensitivity check meaningful here, unlike
the Mo-PAE sample.

`traj_id` = each user's own `.plt` files, dense-indexed in filename-sorted
order. GeoLife's filenames are `YYYYMMDDHHMMSS.plt`, so lexicographic sort
is chronological order -- confirmed against the files' own timestamp column,
not assumed. `cell_id` is NOT set here; tessellation (grid + H3) is applied
generically by scripts/build_dataset.py, same as Foursquare/Gowalla.
"""
from __future__ import annotations

import pathlib

import pandas as pd

HAS_REAL_TIMESTAMP = True

_PLT_COLUMNS = ["lat", "lon", "_unused", "alt_ft", "_date_serial", "date_str", "time_str"]


def _read_plt(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=6, header=None, names=_PLT_COLUMNS,
                        usecols=["lat", "lon", "date_str", "time_str"])


def load_geolife_full_unified(data_dir: str) -> pd.DataFrame:
    data_root = pathlib.Path(data_dir)
    user_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    frames = []
    for user_idx, user_dir in enumerate(user_dirs):
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.is_dir():
            continue
        plt_files = sorted(traj_dir.glob("*.plt"))  # filename = YYYYMMDDHHMMSS, lexicographic == chronological
        for traj_idx, plt_path in enumerate(plt_files):
            pts = _read_plt(plt_path)
            if pts.empty:
                continue
            pts["user_id"] = user_idx + 1
            pts["traj_id"] = traj_idx + 1
            frames.append(pts)

    raw = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(raw["date_str"] + " " + raw["time_str"])

    df = pd.DataFrame()
    df["user_id"] = raw["user_id"]
    df["traj_id"] = raw["traj_id"]
    df["ts"] = ts
    df["lat"] = raw["lat"].astype(float)
    df["lon"] = raw["lon"].astype(float)
    df["weekday"] = ts.dt.weekday
    df["hour"] = ts.dt.hour

    return df[["user_id", "traj_id", "ts", "lat", "lon", "weekday", "hour"]]
