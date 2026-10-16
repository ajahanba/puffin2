"""Gowalla check-ins (Stanford SNAP) -> unified schema.

Raw format confirmed directly from the downloaded file (no header,
tab-separated): user_id, check-in time (ISO8601 UTC, e.g.
"2010-10-19T23:55:27Z"), lat, lon, location_id. Confirmed count: 6,442,892
rows.

Real lat/lon and a real UTC timestamp are both present (HAS_REAL_TIMESTAMP =
True). `cell_id` is NOT set here; tessellation is applied generically by
scripts/build_dataset.py, same as Foursquare. There is no native trajectory
id, so each user's check-ins are segmented into daily trajectories by UTC
date (Gowalla gives no per-checkin timezone offset, unlike Foursquare, so
this is UTC date rather than local date -- a day boundary near midnight UTC
can therefore split what was really one local-time outing; documented here
rather than silently assumed away).

The plan's minimum target is "Gowalla-US"; `us_only=True` (default) filters
to a continental-US bounding box (lat 24-50 N, lon -125 to -66 W) -- an
approximation, not a real reverse-geocode, logged in the returned dataframe's
row count vs the raw 6,442,892 so the drop is visible in stats.json.
"""
from __future__ import annotations

import pandas as pd

from ._checkin_utils import assign_daily_traj_id

HAS_REAL_TIMESTAMP = True

US_BBOX = dict(lat_min=24.0, lat_max=50.0, lon_min=-125.0, lon_max=-66.0)


def load_gowalla_unified(path: str, us_only: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", header=None,
                       names=["raw_user_id", "ts_str", "lat", "lon", "location_id"])

    if us_only:
        raw = raw[raw["lat"].between(US_BBOX["lat_min"], US_BBOX["lat_max"])
                   & raw["lon"].between(US_BBOX["lon_min"], US_BBOX["lon_max"])]

    ts = pd.to_datetime(raw["ts_str"], utc=True).dt.tz_localize(None)
    user_dict = {old: new + 1 for new, old in enumerate(pd.unique(raw["raw_user_id"]))}

    df = pd.DataFrame()
    df["user_id"] = raw["raw_user_id"].map(user_dict)
    df["ts"] = ts
    df["lat"] = raw["lat"].astype(float)
    df["lon"] = raw["lon"].astype(float)
    df["weekday"] = ts.dt.weekday
    df["hour"] = ts.dt.hour
    df["traj_id"] = assign_daily_traj_id(df, ts.dt.date)

    return df[["user_id", "traj_id", "ts", "lat", "lon", "weekday", "hour"]]
