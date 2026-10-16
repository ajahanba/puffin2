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

Per the PI's direction after the first Gowalla-US build: metro-level subsets
(NYC + a few other large US metros) are the main experimental unit going
forward; full-US Gowalla is kept only for the Phase 8 scalability study.
`bbox` selects which: "us" (the original continental-US box, lat 24-50 N,
lon -125 to -66 W), a metro name from METRO_BBOXES below, or None for no
spatial filtering at all (every row). All boxes are plain lat/lon rectangles,
not real reverse-geocoded city boundaries -- approximations, logged in the
returned dataframe's row count vs the raw 6,442,892 so the drop from each is
visible in stats.json.
"""
from __future__ import annotations

import pandas as pd

from ._checkin_utils import assign_daily_traj_id

HAS_REAL_TIMESTAMP = True

US_BBOX = dict(lat_min=24.0, lat_max=50.0, lon_min=-125.0, lon_max=-66.0)

# Approximate metro-area bounding boxes (plain lat/lon rectangles, not real
# city/metro-area polygons). NYC is the plan's named example; LA, Chicago and
# the SF Bay Area are large US metros with plausible 2009-2010 Gowalla usage,
# chosen to give geographic spread (coasts + midwest), not from any special
# selection criterion -- revisit if a dataset-specific reason favors others.
METRO_BBOXES = dict(
    nyc=dict(lat_min=40.40, lat_max=41.00, lon_min=-74.30, lon_max=-73.60),
    la=dict(lat_min=33.60, lat_max=34.40, lon_min=-118.90, lon_max=-117.60),
    chicago=dict(lat_min=41.60, lat_max=42.10, lon_min=-88.00, lon_max=-87.50),
    sf_bay=dict(lat_min=37.20, lat_max=38.00, lon_min=-122.60, lon_max=-121.80),
)


def load_gowalla_unified(path: str, bbox: str | dict | None = "us") -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", header=None,
                       names=["raw_user_id", "ts_str", "lat", "lon", "location_id"])

    if isinstance(bbox, str):
        bbox = US_BBOX if bbox == "us" else METRO_BBOXES[bbox]
    if bbox is not None:
        raw = raw[raw["lat"].between(bbox["lat_min"], bbox["lat_max"])
                   & raw["lon"].between(bbox["lon_min"], bbox["lon_max"])]

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
