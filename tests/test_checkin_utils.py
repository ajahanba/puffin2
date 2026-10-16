#!/usr/bin/env python
"""Regression test for the index-misalignment bug in assign_daily_traj_id
(fixed 2026-10-15): a caller dataframe with a non-contiguous index (e.g.
after Gowalla's US bounding-box filter drops rows) must not corrupt the
result. This exact bug shape silently NaN-filled ~32% of Gowalla's rows
before being caught -- see CHANGES_FROM_ORIGINAL.md for the full account.

No pytest dependency -- run directly: `python tests/test_checkin_utils.py`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.data.datasets._checkin_utils import assign_daily_traj_id


def test_non_contiguous_index():
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 2],
        "date_str": ["2020-01-01", "2020-01-01", "2020-01-02",
                     "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
    })
    df = df.drop(index=[1, 4]).copy()  # simulate a row-dropping filter
    assert list(df.index) == [0, 2, 3, 5, 6], "test setup should leave a non-contiguous index"

    local_date = pd.to_datetime(df["date_str"]).dt.date
    result = assign_daily_traj_id(df, local_date)

    assert not pd.isna(result).any(), "traj_id must never be null regardless of df's index"
    expected = np.array([1, 2, 1, 2, 3])  # user1: day1,day2 ; user2: day1,day2,day3
    assert np.array_equal(result, expected), f"expected {expected}, got {result}"
    print("OK: assign_daily_traj_id is correct under a non-contiguous index")


def test_contiguous_index_unaffected():
    df = pd.DataFrame({"user_id": [1, 1, 2], "date_str": ["2020-01-01", "2020-01-02", "2020-01-01"]})
    local_date = pd.to_datetime(df["date_str"]).dt.date
    result = assign_daily_traj_id(df, local_date)
    assert np.array_equal(result, np.array([1, 2, 1]))
    print("OK: assign_daily_traj_id unaffected on the normal contiguous-index case")


if __name__ == "__main__":
    test_non_contiguous_index()
    test_contiguous_index_unaffected()
    print("All index-safety regression tests passed.")
