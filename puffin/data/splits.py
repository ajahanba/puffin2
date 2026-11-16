"""Two independent splits, added as columns to the unified dataset.

`temporal_split`: per user, first `train_frac` of their own timeline -> train,
the rest -> test. Used for utility and (within-split) re-id evaluation.

`membership_split`: for membership inference. Two orthogonal cuts:
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
    responsible for setting that correctly and should say so in stats.json.

    Vectorized (one global sort + groupby.cumcount/transform, both
    C-level) rather than a Python-level loop over every user, which keeps
    this tractable on YJMob100K's 100,000 users."""
    df = df.copy()
    sorted_df = df.sort_values(["user_id", "ts"])
    rank_in_user = sorted_df.groupby("user_id").cumcount().to_numpy()
    count_per_user = sorted_df.groupby("user_id")["user_id"].transform("count").to_numpy()
    cut = (train_frac * count_per_user).astype(int)
    split_vals = np.where(rank_in_user < cut, "train", "test")
    df["split"] = pd.Series(split_vals, index=sorted_df.index).reindex(df.index)
    return df


def membership_split(df: pd.DataFrame, nonmember_traj_frac: float = 0.2,
                      user_disjoint_frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Vectorized (one groupby + one merge, no per-user loop, O(n_rows)
    rather than O(n_users * n_rows)) so this stays tractable on
    YJMob100K's scale (100,000 users x 111.5M rows)."""
    df = df.copy()
    rng = np.random.default_rng(seed)

    all_users = df["user_id"].unique()
    n_disjoint = max(1, int(user_disjoint_frac * len(all_users)))
    disjoint_users = set(rng.choice(all_users, size=n_disjoint, replace=False))
    df["user_disjoint"] = df["user_id"].isin(disjoint_users)

    traj_df = df.loc[~df["user_disjoint"], ["user_id", "traj_id"]].drop_duplicates().copy()
    traj_df["r"] = rng.random(len(traj_df))
    traj_df = traj_df.sort_values(["user_id", "r"])
    rank_in_user = traj_df.groupby("user_id").cumcount()
    n_trajs = traj_df.groupby("user_id")["traj_id"].transform("count")
    n_nonmember = (nonmember_traj_frac * n_trajs).round().astype(int)
    traj_df["is_nonmember_traj"] = rank_in_user < n_nonmember

    df = df.merge(traj_df[["user_id", "traj_id", "is_nonmember_traj"]],
                   on=["user_id", "traj_id"], how="left")
    df["is_nonmember_traj"] = df["is_nonmember_traj"].fillna(False).astype(bool)
    df["mi_split"] = np.where(df["is_nonmember_traj"] | df["user_disjoint"], "nonmember", "member")
    return df.drop(columns=["is_nonmember_traj"])
