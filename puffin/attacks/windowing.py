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


def count_sl_coverage(df: pd.DataFrame, sl: int, respect_traj_boundary: bool) -> dict:
    """How many users/windows a given SL and windowing mode actually yield,
    per split -- computed via groupby sizes, without materializing window
    arrays (cheap even at GeoLife's 24M-window scale). Used to report the
    coverage trade-off between SL=10 (main) and SL=5 (sensitivity), and
    between per-trajectory and full-sequence windowing."""
    out = {}
    for split_name in ("train", "test"):
        split_df = df[df["split"] == split_name]
        group_cols = ["user_id", "traj_id"] if respect_traj_boundary else ["user_id"]
        sizes = split_df.groupby(group_cols, sort=False).size()
        windows_per_group = (sizes - sl).clip(lower=0)
        if respect_traj_boundary:
            windows_per_user = windows_per_group.groupby(level=0).sum()
        else:
            windows_per_user = windows_per_group
        out[split_name] = dict(
            total_windows=int(windows_per_group.sum()),
            users_with_ge1_window=int((windows_per_user >= 1).sum()),
            users_total=int(split_df["user_id"].nunique()),
        )
    return out


def build_sl_windows(df: pd.DataFrame, sl: int = 10, max_windows_per_user: int | None = None,
                      seed: int = 0, respect_traj_boundary: bool = True):
    """`respect_traj_boundary=True` (default, use for GPS data like GeoLife,
    where traj_id is a real separate trip): windows never cross a
    (user_id, traj_id) boundary. `respect_traj_boundary=False` (use for
    check-in data like Foursquare/Gowalla): windows are built over each
    user's FULL chronological sequence within the split, ignoring the
    traj_id boundary -- a PI correction after Checkpoint 2 (see
    CHANGES_FROM_ORIGINAL.md). Foursquare/Gowalla's traj_id is only a
    same-calendar-day grouping invented by `_checkin_utils.py` because
    those datasets have no native trajectory id; respecting it as if it
    meant something to the user's actual behavior left only 71/1083
    Foursquare users with even one SL=10 window. Windows still never cross
    the train/test split boundary either way.

    Returns dict with 'train' and 'test' keys, each holding:
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
    [0, total_users-1]; the dataset's own user_id is 1-indexed);
    ts: (N,) array, the window's OWN FIRST point's timestamp -- added so a
    caller can do a chronological (not random) validation split for early
    stopping, per the PI's Checkpoint 2c request (puffin.attacks.reid's
    `_train_sequence_attacker` sorts on this).

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
        split_df = df[df["split"] == split_name]
        # PI-caught performance bug (2026-11-09, found via a live py-spy
        # stack trace on a stuck TrajGAN job): pandas' NDFrame.__finalize__
        # deepcopies a non-empty .attrs dict on every slice/groupby
        # operation. Callers routinely pass a df carrying
        # assign_grid_cells's grid_cell_mapping (tens of thousands of
        # entries) in .attrs, and _windows_from_frame's nested per-user/
        # per-trajectory groupby loop below pays a full deepcopy of that
        # dict on every group -- this (not genuine windowing work) is
        # almost certainly the real explanation for several "unexplained"
        # multi-hour slowdowns across this project's history. Clearing
        # attrs here makes every downstream groupby a no-op check instead.
        split_df.attrs = {}
        out[split_name] = _windows_from_frame(split_df, sl, max_windows_per_user, rng, respect_traj_boundary)
    return out


def _windows_from_frame(frame: pd.DataFrame, sl: int, max_windows_per_user, rng, respect_traj_boundary: bool):
    """Shared window-building loop used by both build_sl_windows (one call
    per pre-existing split) and build_sl_windows_contemporaneous (one call
    on each user's own randomly-partitioned pool)."""
    cell_rows, wd_rows, hr_rows, next_rows, user_rows, ts_rows = [], [], [], [], [], []
    for user_id, user_g in frame.groupby("user_id", sort=False):
        user_cell, user_wd, user_hr, user_next, user_ts = [], [], [], [], []
        groups = (user_g.groupby("traj_id", sort=False) if respect_traj_boundary
                  else [(None, user_g)])
        for _traj_id, g in groups:
            if len(g) <= sl:
                continue
            g = g.sort_values("ts")
            cell = g["cell_id"].to_numpy()
            wd = g["weekday"].to_numpy()
            hr = g["hour"].to_numpy()
            ts = g["ts"].to_numpy()
            for i in range(len(g) - sl):
                user_cell.append(cell[i:i + sl])
                user_wd.append(wd[i:i + sl])
                user_hr.append(hr[i:i + sl])
                user_next.append(cell[i + sl] - 1)
                user_ts.append(ts[i])
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
            ts_rows.append(user_ts[i])
            user_rows.append(user_id - 1)  # schema is 1-indexed; cross_entropy needs 0-indexed classes
    return dict(
        cell=np.array(cell_rows, dtype=np.int64) if cell_rows else np.zeros((0, sl), dtype=np.int64),
        weekday=np.array(wd_rows, dtype=np.int64) if wd_rows else np.zeros((0, sl), dtype=np.int64),
        hour=np.array(hr_rows, dtype=np.int64) if hr_rows else np.zeros((0, sl), dtype=np.int64),
        next_cell=np.array(next_rows, dtype=np.int64) if next_rows else np.zeros((0,), dtype=np.int64),
        user=np.array(user_rows, dtype=np.int64) if user_rows else np.zeros((0,), dtype=np.int64),
        ts=np.array(ts_rows) if ts_rows else np.zeros((0,)),
    )


def build_sl_windows_contemporaneous(df: pd.DataFrame, sl: int = 10, max_windows_per_user: int | None = None,
                                      seed: int = 0, respect_traj_boundary: bool = True, test_frac: float = 0.2):
    """Checkpoint 2c, PI-requested: a (b) is build_sl_windows's existing
    chronological train/test split (Phase 1's temporal_split -- train is
    strictly earlier than test for every user). This is (a), the
    contemporaneous control: ALL of a user's windows (ignoring the `split`
    column entirely, pooling what Phase 1 called train and test together),
    then a RANDOM (not time-based) per-user partition into train/test --
    so there's no forced time gap between what the attacker trains on and
    what it's evaluated against. Comparing disparate-risk results from this
    against build_sl_windows's output isolates how much of the earlier
    attacker val/test gap was genuine behavioral drift over time vs. just
    the general difficulty of predicting held-out windows at all."""
    rng = np.random.default_rng(seed)
    # Same attrs-deepcopy fix as build_sl_windows (see its comment) -- here
    # _windows_from_frame runs on `df` directly (no filter/copy step gives
    # us a safe-to-mutate object), so save/restore the caller's attrs
    # rather than clearing them permanently.
    _saved_attrs, df.attrs = df.attrs, {}
    try:
        all_windows = _windows_from_frame(df, sl, None, rng, respect_traj_boundary)  # no cap yet -- cap after split
    finally:
        df.attrs = _saved_attrs

    n = len(all_windows["user"])
    perm = rng.permutation(n)
    n_test = int(test_frac * n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    def _subset_and_cap(idx):
        sub = {k: v[idx] for k, v in all_windows.items()}
        n_sub = len(sub["user"])
        if max_windows_per_user is None or n_sub == 0:
            return sub
        # cap per user, same semantics as _windows_from_frame's own cap
        keep = []
        for u in np.unique(sub["user"]):
            u_idx = np.where(sub["user"] == u)[0]
            if len(u_idx) > max_windows_per_user:
                u_idx = rng.choice(u_idx, size=max_windows_per_user, replace=False)
            keep.append(u_idx)
        keep = np.concatenate(keep)
        return {k: v[keep] for k, v in sub.items()}

    return {"train": _subset_and_cap(train_idx), "test": _subset_and_cap(test_idx)}
