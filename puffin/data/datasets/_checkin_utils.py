"""Shared helpers for check-in style datasets (Foursquare, Gowalla) that have
no inherent trajectory id. Both segment each user's check-ins into daily
trajectories -- the common convention in the POI/TUL literature -- rather than
inventing a dataset-specific rule twice."""
from __future__ import annotations

import pandas as pd


def assign_daily_traj_id(df: pd.DataFrame, local_date: pd.Series):
    """local_date: one calendar-date value per row (already timezone-adjusted
    by the caller if timezone info is available). Returns a dense,
    1-indexed-per-user traj_id as a plain numpy array (not a Series) so the
    caller's `df[...] = ...` assignment is always positional. A Series would
    carry `key`'s own fresh 0..n-1 index, which silently mismatches (and
    NaN-fills) whenever `df` itself has a non-contiguous index -- e.g. after
    an upstream row filter like Gowalla's US bounding box."""
    key = pd.DataFrame({"user_id": df["user_id"].values, "date": local_date.values})
    traj = key.groupby("user_id")["date"].transform(lambda d: d.rank(method="dense").astype(int))
    return traj.to_numpy()
