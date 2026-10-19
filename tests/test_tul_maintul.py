#!/usr/bin/env python
"""Correctness checks for puffin/attacks/tul_maintul.py (Phase 3 task 4).

Run directly: `python tests/test_tul_maintul.py`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.tul_maintul import (
    build_trajectory_dataset, _sample_long_term, run_tul_maintul,
)


def _toy_traj_df(n_users=15, n_traj_per_user=10, traj_len=12, n_cells=30, seed=0,
                  disjoint_users=2):
    """Each user gets `n_traj_per_user` trajectories of `traj_len` points.
    Each user's trajectories are made RECOGNIZABLE: every point of every
    trajectory for user u is drawn from a small user-specific band of cell
    ids (disjoint across users, with slight overlap) -- so a correct
    learner should be able to do much better than chance, giving the
    end-to-end test below a real, non-flaky signal to check for. A few
    users are marked user_disjoint (entirely held out, never trainable)
    to exercise that exclusion path."""
    rng = np.random.default_rng(seed)
    cells_per_user = max(2, n_cells // n_users)
    rows = []
    ts_counter = 0
    for u in range(1, n_users + 1):
        band_start = ((u - 1) * cells_per_user) % n_cells
        band = [(band_start + k) % n_cells + 1 for k in range(cells_per_user)]
        for traj_id in range(1, n_traj_per_user + 1):
            for t in range(traj_len):
                ts_counter += 1
                rows.append(dict(
                    user_id=u, traj_id=traj_id, ts=pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=ts_counter),
                    lat=0.0, lon=0.0, cell_id=int(rng.choice(band)), weekday=t % 7, hour=t % 24,
                    user_disjoint=(u > n_users - disjoint_users),
                ))
    df = pd.DataFrame(rows)

    # membership_split stand-in: last 2 trajectories per user -> nonmember (test), rest -> member (train).
    df["mi_split"] = "member"
    for u in df["user_id"].unique():
        last_two = sorted(df.loc[df["user_id"] == u, "traj_id"].unique())[-2:]
        df.loc[(df["user_id"] == u) & (df["traj_id"].isin(last_two)), "mi_split"] = "nonmember"
    return df


def test_build_trajectory_dataset_excludes_disjoint_users():
    df = _toy_traj_df(n_users=10, n_traj_per_user=6, traj_len=5, disjoint_users=3)
    data = build_trajectory_dataset(df, max_traj_len=50, min_traj_len=1)
    disjoint_raw_ids = set(df.loc[df["user_disjoint"], "user_id"].unique().tolist())
    assert disjoint_raw_ids.isdisjoint(set(data["user_to_idx"].keys())), \
        "user_disjoint users must never get a classifier slot (closed-world TUL)"
    non_disjoint_ids = set(df.loc[~df["user_disjoint"], "user_id"].unique().tolist())
    assert set(data["user_to_idx"].keys()) == non_disjoint_ids
    print("OK: build_trajectory_dataset excludes user_disjoint users entirely")


def test_build_trajectory_dataset_respects_mi_split_and_truncation():
    df = _toy_traj_df(n_users=5, n_traj_per_user=8, traj_len=20, disjoint_users=0)
    data = build_trajectory_dataset(df, max_traj_len=7, min_traj_len=1)
    for u in data["user_to_idx"]:
        assert len(data["train"][u]) == 6, (u, len(data["train"][u]))  # 8 - 2 nonmember = 6 member
        assert len(data["test"][u]) == 2
        for cell, hour in data["train"][u] + data["test"][u]:
            assert len(cell) == 7, "trajectory must be truncated to max_traj_len"
            assert len(hour) == 7
            assert (cell > 0).all(), "cell ids must stay 1-indexed (0 is reserved for padding)"
            assert ((hour >= 1) & (hour <= 24)).all(), "hour must be shifted into 1..24 (0 reserved for pad)"
    print("OK: build_trajectory_dataset splits member->train/nonmember->test and truncates to max_traj_len")


def test_sample_long_term_never_crashes_on_small_user():
    rng = np.random.default_rng(0)
    # a user with only 1 train trajectory: long_term_num must clamp to 0, not crash.
    one_traj = [(np.array([1, 2, 3]), np.array([1, 2, 3]))]
    cells, hours = _sample_long_term(one_traj, this_idx=0, long_term_num=8, rng=rng)
    assert np.array_equal(cells, one_traj[0][0])
    assert np.array_equal(hours, one_traj[0][1])

    # a user with 5 trajectories and long_term_num=8: clamp to 4 others + itself = 5 trajs concatenated.
    five_trajs = [(np.array([i, i + 1]), np.array([i, i + 1])) for i in range(5)]
    cells, hours = _sample_long_term(five_trajs, this_idx=2, long_term_num=8, rng=rng)
    assert len(cells) == 2 * 5, len(cells)  # all 5 trajectories concatenated, length 2 each
    print("OK: _sample_long_term clamps long_term_num instead of crashing on small users")


def test_end_to_end_tul_links_trajectories_above_chance():
    """The core correctness claim: with user-specific, non-overlapping-ish
    cell-id bands (see _toy_traj_df), a correctly-wired student+teacher
    pair should recover the user with accuracy well above chance
    (1/n_users) on held-out test trajectories -- if the data plumbing
    (padding, masking, label indexing, long-term augmentation) were wrong,
    this would stay near chance instead."""
    torch.manual_seed(0)
    df = _toy_traj_df(n_users=12, n_traj_per_user=10, traj_len=15, n_cells=36, seed=1, disjoint_users=2)
    device = torch.device("cpu")
    per_user_df, summary = run_tul_maintul(
        df, device, max_traj_len=20, min_traj_len=1, long_term_num=4,
        embed_size=32, hidden_size=64, num_layers=1, num_heads=2,
        batch_size=16, max_epochs=25, patience=6, lr=3e-3,
        temperature=5.0, lambda_distill=5.0, val_frac=0.15, seed=0)

    assert summary["n_users"] == 10  # 12 - 2 disjoint
    assert summary["n_test_trajectories"] > 0
    for col in ("user_id", "n_train_trajectories", "n_test_trajectories",
                "student_top1", "student_top5", "teacher_top1", "teacher_top5",
                "student_mean_confidence", "teacher_mean_confidence"):
        assert col in per_user_df.columns, col
    assert len(per_user_df) == summary["n_users"]

    chance = 1.0 / summary["n_users"]
    assert summary["student_overall_top1"] > 3 * chance, summary
    assert summary["teacher_overall_top1"] > 3 * chance, summary
    assert summary["student_overall_top5"] >= summary["student_overall_top1"]

    valid = per_user_df.dropna(subset=["student_top1"])
    assert len(valid) > 0
    print(f"OK: end-to-end TUL links trajectories well above chance "
          f"(chance={chance:.3f}, student top1={summary['student_overall_top1']:.3f}, "
          f"teacher top1={summary['teacher_overall_top1']:.3f})")


if __name__ == "__main__":
    test_build_trajectory_dataset_excludes_disjoint_users()
    test_build_trajectory_dataset_respects_mi_split_and_truncation()
    test_sample_long_term_never_crashes_on_small_user()
    test_end_to_end_tul_links_trajectories_above_chance()
    print("All tul_maintul tests passed.")
