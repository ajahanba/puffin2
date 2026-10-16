"""Phase 1 task 4: drop users with too few observations, and report exactly
how many users/rows that dropped -- never silently."""
from __future__ import annotations

import pandas as pd


def filter_users_by_min_observations(df: pd.DataFrame, min_obs: int) -> tuple[pd.DataFrame, dict]:
    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= min_obs].index
    kept = df[df["user_id"].isin(keep_users)].copy()
    report = dict(
        min_obs=min_obs,
        users_before=int(counts.shape[0]),
        users_after=int(len(keep_users)),
        users_dropped=int(counts.shape[0] - len(keep_users)),
        rows_before=int(len(df)),
        rows_after=int(len(kept)),
        rows_dropped=int(len(df) - len(kept)),
    )
    return kept, report
