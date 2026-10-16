#!/usr/bin/env python
"""Correctness checks for the vectorized rewrites of temporal_split,
membership_split (puffin/data/splits.py) and per_user_entropy
(puffin/data/stats.py). The originals used O(n_users) Python loops / a
per-user .isin() rescan of the whole dataframe; the latter was quadratic
enough to still be running after 90+ minutes on YJMob100K's 100,000 users
and had to be killed -- see CHANGES_FROM_ORIGINAL.md. These checks don't
require bit-exact reproduction of the old RNG consumption order, only that
the *shape* of each result is correct.

Run directly: `python tests/test_splits_stats.py`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.data.splits import temporal_split, membership_split
from puffin.data.stats import per_user_entropy


def test_temporal_split_per_user_cut():
    # user 1: 10 rows -> train_frac=0.8 -> first 8 (by ts) train, last 2 test.
    # user 2: 5 rows -> first 4 train, last 1 test.
    df = pd.DataFrame({
        "user_id": [1] * 10 + [2] * 5,
        "ts": list(range(10)) + list(range(5)),
    })
    out = temporal_split(df, train_frac=0.8)
    u1 = out[out["user_id"] == 1].sort_values("ts")
    u2 = out[out["user_id"] == 2].sort_values("ts")
    assert list(u1["split"]) == ["train"] * 8 + ["test"] * 2, list(u1["split"])
    assert list(u2["split"]) == ["train"] * 4 + ["test"] * 1, list(u2["split"])
    print("OK: temporal_split cuts each user at their own train_frac boundary")


def test_temporal_split_unsorted_input():
    # Rows arrive out of ts order; result must still be sorted logically by ts.
    df = pd.DataFrame({"user_id": [1, 1, 1, 1, 1], "ts": [4, 2, 0, 3, 1]})
    out = temporal_split(df, train_frac=0.6)
    by_ts = out.sort_values("ts")
    assert list(by_ts["split"]) == ["train", "train", "train", "test", "test"]
    print("OK: temporal_split is correct even when input rows aren't ts-sorted")


def test_membership_split_shapes():
    rng_users = list(range(1, 101))  # 100 users
    rows = []
    for u in rng_users:
        for t in range(1, 6):  # 5 trajectories each
            rows.append((u, t))
    df = pd.DataFrame(rows, columns=["user_id", "traj_id"])

    out = membership_split(df, nonmember_traj_frac=0.2, user_disjoint_frac=0.1, seed=0)

    n_disjoint_users = out.loc[out["user_disjoint"], "user_id"].nunique()
    assert n_disjoint_users == 10, n_disjoint_users  # 10% of 100

    # every disjoint user's rows must all be "nonmember"
    disjoint_rows = out[out["user_disjoint"]]
    assert (disjoint_rows["mi_split"] == "nonmember").all()

    # every non-disjoint user should have exactly round(0.2*5)=1 nonmember trajectory
    for u, g in out[~out["user_disjoint"]].groupby("user_id"):
        n_nonmember_trajs = g.loc[g["mi_split"] == "nonmember", "traj_id"].nunique()
        assert n_nonmember_trajs == 1, (u, n_nonmember_trajs)
    print("OK: membership_split gives the right disjoint-user count and per-user nonmember-traj count")


def test_per_user_entropy_matches_manual_calc():
    # user 1: cells [A,A,A,B] -> p=[0.75,0.25] -> entropy = -0.75log2(0.75)-0.25log2(0.25)
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2],
        "cell_id": ["A", "A", "A", "B", "X", "Y"],
    })
    ent = per_user_entropy(df)
    expected_u1 = -0.75 * np.log2(0.75) - 0.25 * np.log2(0.25)
    expected_u2 = -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)  # = 1.0, max entropy for 2 equally likely cells
    assert abs(ent[1] - expected_u1) < 1e-9, (ent[1], expected_u1)
    assert abs(ent[2] - expected_u2) < 1e-9, (ent[2], expected_u2)
    print("OK: per_user_entropy matches a hand-computed Shannon entropy")


if __name__ == "__main__":
    test_temporal_split_per_user_cut()
    test_temporal_split_unsorted_input()
    test_membership_split_shapes()
    test_per_user_entropy_matches_manual_calc()
    print("All splits/stats vectorization correctness tests passed.")
