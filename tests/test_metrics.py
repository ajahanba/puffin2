#!/usr/bin/env python
"""Correctness checks for puffin/metrics/*.py (Phase 3 task 1)."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.metrics.privacy import privacy_gain, protection_at_k
from puffin.metrics.fairness import similar_pairs_within_epsilon, lipschitz_violation, max_min_gain, gini, gain_percentiles
from puffin.metrics.leveling_down import leveling_down_check
from puffin.attacks.analysis import paired_bootstrap_test


def test_privacy_gain_basic_and_zero_division():
    raw = np.array([0.8, 0.5, 0.0])
    post = np.array([0.4, 0.5, 0.0])
    gain = privacy_gain(raw, post)
    assert abs(gain[0] - 0.5) < 1e-9  # halved risk -> 50% gain
    assert abs(gain[1] - 0.0) < 1e-9  # unchanged -> 0% gain
    assert gain[2] == 0.0  # raw=0 -> defined as 0, not NaN
    print("OK: privacy_gain basic + zero-raw-risk case")


def test_protection_at_k_is_privacy_gain():
    a, b = np.array([0.9, 0.2]), np.array([0.3, 0.2])
    assert np.allclose(protection_at_k(a, b), privacy_gain(a, b))
    print("OK: protection_at_k matches privacy_gain")


def test_lipschitz_violation_detects_known_violation():
    # 3 users, gains [0, 1, 0]; distances all 0.1 -> bound with L=1 is 0.1.
    # |gain0-gain1|=1 > 0.1 (violation); |gain1-gain2|=1 > 0.1 (violation); |gain0-gain2|=0 (no violation)
    gain = np.array([0.0, 1.0, 0.0])
    dist = np.full((3, 3), 0.1)
    np.fill_diagonal(dist, 0.0)
    out = lipschitz_violation(gain, dist, L=1.0)
    assert out["n_pairs"] == 3
    assert abs(out["violation_rate"] - 2 / 3) < 1e-9
    assert out["mean_violation_size"] > 0
    print("OK: lipschitz_violation finds the 2 expected violations out of 3 pairs:", out)


def test_similar_pairs_within_epsilon():
    gain = np.array([0.5, 0.52, 0.9])
    dist = np.array([[0, 0.01, 0.5], [0.01, 0, 0.5], [0.5, 0.5, 0]])
    # only pair (0,1) is "close" (d<0.1); their gain diff is 0.02 <= epsilon=0.05 -> within
    frac = similar_pairs_within_epsilon(gain, dist, epsilon=0.05, delta=0.1)
    assert abs(frac - 1.0) < 1e-9
    print("OK: similar_pairs_within_epsilon")


def test_max_min_and_gini_and_percentiles():
    gain_equal = np.array([0.5, 0.5, 0.5, 0.5])
    assert gini(gain_equal) < 1e-9  # perfectly equal -> Gini 0
    gain_unequal = np.array([0.0, 0.0, 0.0, 1.0])
    assert gini(gain_unequal) > 0.5  # highly unequal -> high Gini
    assert max_min_gain(np.array([0.3, -0.1, 0.9])) == -0.1
    p = gain_percentiles(np.arange(1, 101), percentiles=(10, 50, 90))
    assert p[50] == 50.5
    print("OK: gini (equal=0, unequal>0.5), max_min_gain, gain_percentiles")


def test_leveling_down_check_detects_harm_to_well_protected():
    # 4 users: baseline already protects users well (high gain); mechanism
    # HARMS exactly the users baseline protected best -> leveling down.
    idx = ["u1", "u2", "u3", "u4"]
    baseline = pd.Series([0.9, 0.8, 0.1, 0.1], index=idx)
    mechanism = pd.Series([0.5, 0.4, 0.15, 0.15], index=idx)  # u1,u2 harmed; u3,u4 helped slightly
    out = leveling_down_check(mechanism, baseline)
    assert out["frac_harmed"] == 0.5
    assert out["leveling_down"] is True
    print("OK: leveling_down_check correctly flags leveling-down:", out)


def test_paired_bootstrap_test_excludes_zero_for_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.05, size=200)
    b = rng.normal(0.3, 0.05, size=200)  # clearly different from a
    out = paired_bootstrap_test(a, b)
    assert out["ci_excludes_zero"] is True
    assert out["mean_diff"] > 0
    c = a.copy()  # identical -> diff should be exactly 0, CI must include zero
    out2 = paired_bootstrap_test(a, c)
    assert out2["ci_excludes_zero"] is False
    print("OK: paired_bootstrap_test distinguishes a real difference from none")


if __name__ == "__main__":
    test_privacy_gain_basic_and_zero_division()
    test_protection_at_k_is_privacy_gain()
    test_lipschitz_violation_detects_known_violation()
    test_similar_pairs_within_epsilon()
    test_max_min_and_gini_and_percentiles()
    test_leveling_down_check_detects_harm_to_well_protected()
    test_paired_bootstrap_test_excludes_zero_for_real_difference()
    print("All metrics tests passed.")
