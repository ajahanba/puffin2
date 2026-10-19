#!/usr/bin/env python
"""Correctness checks for puffin/attacks/mia_inhouse.py (Phase 3 task 3).

Run directly: `python tests/test_mia_inhouse.py`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.mia_inhouse import (
    _windows_by_membership, _best_threshold, run_loss_threshold_mia,
)


def _toy_membership_df(n_users=20, member_len=15, nonmember_len=15, n_cells=25, seed=0):
    """Each user gets exactly two trajectories: traj 1 is "member" (the
    target model trains on it), traj 2 is "nonmember" (held out). Cell
    sequences are a fixed, random-but-reproducible walk over a small
    vocabulary so a small model can plausibly memorize the member windows
    within a handful of epochs."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        for traj_id, length, mi_split in ((1, member_len, "member"), (2, nonmember_len, "nonmember")):
            cells = rng.integers(1, n_cells + 1, size=length)
            for t, cell in enumerate(cells):
                rows.append(dict(user_id=u, traj_id=traj_id, ts=t, lat=0.0, lon=0.0,
                                  cell_id=int(cell), weekday=t % 7, hour=t % 24,
                                  mi_split=mi_split, user_disjoint=False))
    return pd.DataFrame(rows)


def test_windows_by_membership_respects_mi_split():
    df = _toy_membership_df(n_users=5, member_len=8, nonmember_len=8, n_cells=10)
    sl = 3
    windows = _windows_by_membership(df, sl=sl, max_windows_per_user=None, seed=0)
    # each user: member traj length 8 -> 8-3=5 windows; nonmember traj length 8 -> 5 windows.
    assert len(windows["train"]["cell"]) == 5 * 5, len(windows["train"]["cell"])  # "train" == member
    assert len(windows["test"]["cell"]) == 5 * 5, len(windows["test"]["cell"])    # "test" == nonmember
    # windows must never mix member and nonmember rows: every "train" window's
    # next_cell must come from a row that really was "member" -- checked indirectly
    # by confirming train/test window counts don't overlap with the wrong traj's length
    # (a cross-membership leak would change these counts, since traj 1 and traj 2
    # have different lengths in a variant of this test -- here they're deliberately
    # equal, so also check user coverage instead).
    assert set(windows["train"]["user"].tolist()) == set(range(5))  # 0-indexed
    assert set(windows["test"]["user"].tolist()) == set(range(5))
    print("OK: _windows_by_membership maps member->train, nonmember->test, per user")


def test_windows_by_membership_leak_check_with_unequal_lengths():
    # member traj longer than nonmember -> if the two were ever mixed, the
    # window counts below would not match this exact arithmetic.
    df = _toy_membership_df(n_users=3, member_len=10, nonmember_len=6, n_cells=10)
    sl = 2
    windows = _windows_by_membership(df, sl=sl, max_windows_per_user=None, seed=0)
    assert len(windows["train"]["cell"]) == 3 * (10 - sl), len(windows["train"]["cell"])
    assert len(windows["test"]["cell"]) == 3 * (6 - sl), len(windows["test"]["cell"])
    print("OK: member/nonmember windows aren't mixed even when trajectory lengths differ")


def test_best_threshold_separates_clean_signal():
    rng = np.random.default_rng(0)
    member_scores = rng.normal(0.9, 0.03, size=200)
    nonmember_scores = rng.normal(0.1, 0.03, size=200)
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    t = _best_threshold(scores, labels)
    pred = (scores >= t).astype(int)
    acc = (pred == labels).mean()
    assert acc > 0.95, acc
    assert 0.1 < t < 0.9, t
    print(f"OK: _best_threshold finds a near-perfect cut ({acc:.3f} accuracy) on clean, separated scores")


def test_best_threshold_handles_single_class_calib():
    # degenerate calib split (e.g. a user with only member or only nonmember
    # windows selected into calib by chance) must not crash.
    t = _best_threshold(np.array([0.5, 0.6, 0.7]), np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(t)
    print("OK: _best_threshold doesn't crash on a single-class calibration split")


def test_end_to_end_mia_detects_overfitting_signal():
    """The core correctness claim: a model trained only on "member" windows
    should end up more confident on those exact windows than on windows it
    never trained on ("nonmember") -- the textbook Yeom/Shokri asymmetry.
    Uses a small vocabulary and enough epochs that the model can plausibly
    memorize the (small) member set, so this isn't a flaky near-chance
    result."""
    torch.manual_seed(0)
    df = _toy_membership_df(n_users=25, member_len=15, nonmember_len=15, n_cells=30, seed=1)
    device = torch.device("cpu")
    per_user_df, summary = run_loss_threshold_mia(
        df, device, sl=5, max_windows_per_user=None, seed=0,
        epochs=120, batch_size=64, lr=2e-3, calib_frac=0.5)

    assert len(per_user_df) > 0
    for col in ("user_id", "n_member_windows", "n_nonmember_windows",
                "member_mean_confidence", "nonmember_mean_confidence", "attack_accuracy"):
        assert col in per_user_df.columns, col

    assert summary["n_member_eval"] > 0 and summary["n_nonmember_eval"] > 0
    assert np.isfinite(summary["overall_auc"])
    assert summary["overall_auc"] > 0.5, summary  # better than a coin flip
    assert summary["overall_accuracy"] > 0.5, summary

    valid = per_user_df.dropna(subset=["member_mean_confidence", "nonmember_mean_confidence"])
    assert len(valid) > 0
    mean_gap = (valid["member_mean_confidence"] - valid["nonmember_mean_confidence"]).mean()
    assert mean_gap > 0, mean_gap  # member windows are, on average, more confidently predicted
    print(f"OK: end-to-end MIA detects the overfitting signal "
          f"(overall AUC={summary['overall_auc']:.3f}, mean confidence gap={mean_gap:.3f})")


if __name__ == "__main__":
    test_windows_by_membership_respects_mi_split()
    test_windows_by_membership_leak_check_with_unequal_lengths()
    test_best_threshold_separates_clean_signal()
    test_best_threshold_handles_single_class_calib()
    test_end_to_end_mia_detects_overfitting_signal()
    print("All mia_inhouse tests passed.")
