#!/usr/bin/env python
"""Correctness checks for the recalibrated puffin/metrics/leveling_down.py."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.metrics.leveling_down import leveling_down_check, seed_to_seed_null


def test_seed_to_seed_null_is_centered_at_zero_for_noise_only_seeds():
    rng = np.random.default_rng(0)
    seed_cols = [rng.normal(0.3, 0.02, size=200) for _ in range(3)]
    null = seed_to_seed_null(seed_cols)
    assert len(null) == 200 * 3 * 2  # all ordered pairs, i != j
    assert abs(null.mean()) < 0.01  # no real effect between seeds -> null centered at 0
    print(f"OK: seed-to-seed null from 3 noise-only seeds is centered at {null.mean():+.4f} (expected ~0)")


def test_no_real_effect_does_not_flag_leveling_down():
    """mechanism == baseline in expectation, both just resampled noise --
    should not spuriously classify many users as harmed/helped relative to
    a null built from the SAME noise process."""
    rng = np.random.default_rng(1)
    n = 300
    idx = [f"u{i}" for i in range(n)]
    baseline_seeds = [pd.Series(rng.normal(0.3, 0.02, size=n), index=idx) for _ in range(3)]
    null = seed_to_seed_null(baseline_seeds)
    post_baseline = baseline_seeds[0]
    post_mechanism = pd.Series(rng.normal(0.3, 0.02, size=n), index=idx)  # same distribution, independent draw
    out = leveling_down_check(post_mechanism, post_baseline, null, n_boot=1000)
    assert not out["harmed_rate_exceeds_null"]
    assert out["leveling_down_pattern"] is False
    print(f"OK: no real effect -> frac_harmed={out['frac_harmed']:.3f}, "
          f"harmed_rate_exceeds_null={out['harmed_rate_exceeds_null']} (correctly not flagged)")


def test_real_leveling_down_pattern_is_detected():
    """Plant a real effect: the mechanism raises risk (harms) specifically
    for users who were already well-protected under baseline (low
    baseline risk) -- the leveling-down signature -- while leaving
    already-poorly-protected (high baseline risk) users alone."""
    rng = np.random.default_rng(2)
    n = 200
    idx = [f"u{i}" for i in range(n)]
    baseline_seeds = [pd.Series(rng.normal(0.3, 0.015, size=n), index=idx) for _ in range(3)]
    null = seed_to_seed_null(baseline_seeds)
    post_baseline = pd.Series(np.concatenate([rng.normal(0.1, 0.01, size=100),   # well-protected half
                                               rng.normal(0.5, 0.01, size=100)]),  # poorly-protected half
                               index=idx)
    # mechanism harms (raises risk by 0.15) only the well-protected half
    post_mechanism = post_baseline.copy()
    post_mechanism.iloc[:100] = post_mechanism.iloc[:100] + 0.15
    out = leveling_down_check(post_mechanism, post_baseline, null, n_boot=1000)
    assert out["harmed_rate_exceeds_null"]
    assert out["leveling_down_pattern"] is True
    assert out["frac_harmed"] > 0.3
    print(f"OK: planted leveling-down signal detected -- frac_harmed={out['frac_harmed']:.3f}, "
          f"leveling_down_pattern={out['leveling_down_pattern']}")


def test_thin_null_is_flagged_not_silently_used():
    post_a = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    post_b = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    out = leveling_down_check(post_a, post_b, np.array([0.01, 0.02]))  # null too small
    assert "error" in out
    print("OK: a too-small null (<10 points) is flagged as an error, not silently used for thresholds")


if __name__ == "__main__":
    test_seed_to_seed_null_is_centered_at_zero_for_noise_only_seeds()
    test_no_real_effect_does_not_flag_leveling_down()
    test_real_leveling_down_pattern_is_detected()
    test_thin_null_is_flagged_not_silently_used()
    print("All leveling_down tests passed.")
