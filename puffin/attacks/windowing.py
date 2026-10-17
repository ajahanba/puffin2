"""Phase 2/3 shared: build length-SL windows from a Phase 1 unified parquet
that already has a `split` column (puffin.data.splits.temporal_split).

Windows never cross a (user_id, traj_id) boundary, and never cross the
train/test boundary either -- a window is built entirely from rows sharing
the same split value, so the same leakage issue Phase 0 audited in
mopae.py's original windowing (issue #2/#3) can't recur here. This mirrors
puffin.data.geolife.fixed_windows but is dataset-agnostic (works off the
unified schema, not GeoLife's raw columns) and reports per-user window
counts, consistent with puffin.data.stats.windows_at_sl.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_sl_windows(df: pd.DataFrame, sl: int = 10, max_windows_per_user: int | None = None,
                      seed: int = 0):
    """Returns dict with 'train' and 'test' keys, each holding:
    cell/weekday/hour: (N, sl) int arrays (the window itself);
    next_cell: (N,) int array, 0-indexed (same reason as `user` below --
    it's a cross_entropy target, not an embedding input; `cell`/`weekday`/
    `hour` stay 1-indexed since FixedEncoder's embeddings are sized
    total_locations+1 specifically to accept 1-indexed ids directly), the
    TRUE next location one step past the window (same convention as
    puffin.data.geolife.fixed_windows) -- needed
    so Phase 2's Mo-PAE-fixed training has a genuine next-location utility
    target, not a trivial stand-in (an earlier version used the window's own
    last cell as "the target," which would have let the encoder satisfy the
    utility loss almost for free and distorted the privacy/utility trade-off
    being measured);
    user: (N,) int array, 0-indexed (cross_entropy needs classes in
    [0, total_users-1]; the dataset's own user_id is 1-indexed).

    Built only from that split's rows -- a window never crosses a
    (user_id, traj_id) boundary or the train/test boundary.
    `max_windows_per_user` randomly subsamples each user's windows -- summed
    across ALL of that user's trajectories, within each split -- down to at
    most that many. (An earlier version applied the cap per (user, traj)
    group instead of per user: on GeoLife, where a single user can have
    100+ trajectories, that let a 50-per-group cap produce tens of
    thousands of windows for one user instead of the intended 50, which is
    what caused an OOM during attacker evaluation -- see
    CHANGES_FROM_ORIGINAL.md.) Phase 2 is a motivation-level analysis, not a
    final Phase 8 production run, and GeoLife alone has 24.7M SL=10
    windows; capping keeps attacker training tractable without changing
    which users are represented (every user with >=1 window keeps at least
    one)."""
    rng = np.random.default_rng(seed)
    out = {}
    for split_name in ("train", "test"):
        cell_rows, wd_rows, hr_rows, next_rows, user_rows = [], [], [], [], []
        split_df = df[df["split"] == split_name]
        for user_id, user_g in split_df.groupby("user_id", sort=False):
            user_cell, user_wd, user_hr, user_next = [], [], [], []
            for _traj_id, g in user_g.groupby("traj_id", sort=False):
                if len(g) <= sl:
                    continue
                g = g.sort_values("ts")
                cell = g["cell_id"].to_numpy()
                wd = g["weekday"].to_numpy()
                hr = g["hour"].to_numpy()
                for i in range(len(g) - sl):
                    user_cell.append(cell[i:i + sl])
                    user_wd.append(wd[i:i + sl])
                    user_hr.append(hr[i:i + sl])
                    user_next.append(cell[i + sl] - 1)
            if not user_cell:
                continue
            idx = np.arange(len(user_cell))
            if max_windows_per_user is not None and len(idx) > max_windows_per_user:
                idx = rng.choice(idx, size=max_windows_per_user, replace=False)
            for i in idx:
                cell_rows.append(user_cell[i])
                wd_rows.append(user_wd[i])
                hr_rows.append(user_hr[i])
                next_rows.append(user_next[i])
                user_rows.append(user_id - 1)  # schema is 1-indexed; cross_entropy needs 0-indexed classes
        out[split_name] = dict(
            cell=np.array(cell_rows, dtype=np.int64) if cell_rows else np.zeros((0, sl), dtype=np.int64),
            weekday=np.array(wd_rows, dtype=np.int64) if wd_rows else np.zeros((0, sl), dtype=np.int64),
            hour=np.array(hr_rows, dtype=np.int64) if hr_rows else np.zeros((0, sl), dtype=np.int64),
            next_cell=np.array(next_rows, dtype=np.int64) if next_rows else np.zeros((0,), dtype=np.int64),
            user=np.array(user_rows, dtype=np.int64) if user_rows else np.zeros((0,), dtype=np.int64),
        )
    return out
