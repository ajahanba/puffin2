"""Phase 1 task 3: two independent splits, added as columns so a single
processed file can serve every later phase without re-splitting.

`temporal_split`: per user, first `train_frac` of their own timeline -> train,
the rest -> test. Used for utility and (within-split) re-id evaluation.

`membership_split`: for Phase 3's membership inference. Two orthogonal cuts:
  - `mi_split` in {"member", "nonmember"}: whole TRAJECTORIES held out per
    user (the model never trains on a "nonmember" trajectory for that user).
  - `user_disjoint`: True for an entirely separate set of users whose
    trajectories never appear in training at all (a stronger, user-level
    non-membership notion than per-trajectory holdout).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8, has_real_timestamp: bool = True) -> pd.DataFrame:
    """Adds a `split` column ("train"/"test"). Sorts by `ts` within each
    (user_id, traj_id) group -- if the dataset has no real timestamp
    (`has_real_timestamp=False`), `ts` must already be a monotonic surrogate
    (e.g. original row order) rather than a wall-clock time; the loader is
    responsible for setting that correctly and should say so in stats.json."""
    df = df.copy()
    df["split"] = "train"
    for _, idx in df.groupby("user_id").groups.items():
        g = df.loc[idx].sort_values("ts")
        cut = int(train_frac * len(g))
        test_idx = g.index[cut:]
        df.loc[test_idx, "split"] = "test"
    return df


def membership_split(df: pd.DataFrame, nonmember_traj_frac: float = 0.2,
                      user_disjoint_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)

    all_users = sorted(df["user_id"].unique())
    n_disjoint = max(1, int(user_disjoint_frac * len(all_users)))
    disjoint_users = set(rng.choice(all_users, size=n_disjoint, replace=False))
    df["user_disjoint"] = df["user_id"].isin(disjoint_users)

    df["mi_split"] = "member"
    member_users = [u for u in all_users if u not in disjoint_users]
    for u in member_users:
        user_trajs = sorted(df.loc[df["user_id"] == u, "traj_id"].unique())
        n_nonmember = max(0, round(nonmember_traj_frac * len(user_trajs)))
        if n_nonmember == 0:
            continue
        nonmember_trajs = set(rng.choice(user_trajs, size=n_nonmember, replace=False))
        mask = (df["user_id"] == u) & (df["traj_id"].isin(nonmember_trajs))
        df.loc[mask, "mi_split"] = "nonmember"
    # user-disjoint users are nonmembers by construction (model never trains on them)
    df.loc[df["user_disjoint"], "mi_split"] = "nonmember"
    return df
