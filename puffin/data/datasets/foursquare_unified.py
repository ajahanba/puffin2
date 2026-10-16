"""Foursquare NYC/Tokyo check-ins (Yang et al., IEEE TSMC 2015) -> unified schema.

Raw format confirmed directly from the downloaded file (no header, tab-separated):
user_id, venue_id, venue_category_id, venue_category_name, lat, lon,
utc_offset_minutes, utc_time ("Tue Apr 03 18:00:09 +0000 2012" style).

Real lat/lon and a real UTC timestamp are both present (HAS_REAL_TIMESTAMP =
True), unlike GeoLife's sample. `cell_id` is NOT set here -- this dataset has
real coordinates, so tessellation (grid or H3) is applied generically by
scripts/build_dataset.py via puffin.data.tessellation, the same as Gowalla,
rather than duplicating grid logic per loader. There is no native trajectory
id; each user's check-ins are segmented into daily trajectories using the
*local* date (utc_time + utc_offset_minutes), per `_checkin_utils`.
"""
from __future__ import annotations

import pandas as pd

from ._checkin_utils import assign_daily_traj_id

HAS_REAL_TIMESTAMP = True

_COLUMNS = ["raw_user_id", "venue_id", "venue_category_id", "venue_category_name",
            "lat", "lon", "utc_offset_min", "utc_time"]


def load_foursquare_unified(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", header=None, names=_COLUMNS, encoding="latin-1")

    utc = pd.to_datetime(raw["utc_time"], format="%a %b %d %H:%M:%S %z %Y")
    local_ts = utc.dt.tz_localize(None) + pd.to_timedelta(raw["utc_offset_min"], unit="m")

    user_dict = {old: new + 1 for new, old in enumerate(pd.unique(raw["raw_user_id"]))}

    df = pd.DataFrame()
    df["user_id"] = raw["raw_user_id"].map(user_dict)
    df["ts"] = local_ts
    df["lat"] = raw["lat"].astype(float)
    df["lon"] = raw["lon"].astype(float)
    df["weekday"] = local_ts.dt.weekday
    df["hour"] = local_ts.dt.hour
    df["traj_id"] = assign_daily_traj_id(df, local_ts.dt.date)

    return df[["user_id", "traj_id", "ts", "lat", "lon", "weekday", "hour"]]
