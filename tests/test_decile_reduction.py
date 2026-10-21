#!/usr/bin/env python
"""Correctness checks for puffin/metrics/decile_reduction.py."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.metrics.decile_reduction import decile_reduction_ci, residual_risk_disparity


def test_absolute_reduction_matches_manual_decile_means():
    rng = np.random.default_rng(0)
    deciles = np.repeat(np.arange(3), 50)
    raw = rng.uniform(0.3, 0.6, size=150)
    post = raw - 0.1  # a clean, known absolute reduction of 0.1 everywhere
    out = decile_reduction_ci(raw, post, deciles, n_boot=500)
    for d in range(3):
        manual_raw_mean = raw[deciles == d].mean()
        manual_post_mean = post[deciles == d].mean()
        assert abs(out[d]["raw_mean"] - manual_raw_mean) < 1e-9
        assert abs(out[d]["post_mean"] - manual_post_mean) < 1e-9
        assert abs(out[d]["absolute_reduction"]["mean"] - 0.1) < 0.02
        assert out[d]["absolute_reduction"]["ci_excludes_zero"]
    print("OK: absolute reduction matches manual decile means, CI excludes zero for a real effect")


def test_relative_reduction_handles_known_ratio():
    deciles = np.zeros(100, dtype=int)
    raw = np.full(100, 0.5)
    post = np.full(100, 0.25)  # exactly 50% relative reduction
    out = decile_reduction_ci(raw, post, deciles, n_boot=500)
    assert abs(out[0]["relative_reduction"]["mean"] - 0.5) < 1e-6
    print(f"OK: relative reduction on a known 50% case gives {out[0]['relative_reduction']['mean']:.4f}")


def test_residual_risk_disparity_known_ratio():
    rng = np.random.default_rng(1)
    deciles = np.concatenate([np.zeros(80, dtype=int), np.full(80, 9, dtype=int)])
    post = np.concatenate([rng.normal(0.4, 0.03, size=80), rng.normal(0.1, 0.02, size=80)])  # ratio ~4.0, with noise
    out = residual_risk_disparity(post, deciles, low_decile=0, high_decile=9, n_boot=1000)
    assert abs(out["point_estimate"] - 4.0) < 0.5
    assert out["ci_lo"] < out["point_estimate"] < out["ci_hi"]
    print(f"OK: residual-risk disparity on a known ~4.0x case gives point={out['point_estimate']:.3f}, "
          f"CI=[{out['ci_lo']:.3f},{out['ci_hi']:.3f}]")


def test_near_zero_raw_risk_does_not_crash_and_is_excluded_from_relative():
    deciles = np.zeros(20, dtype=int)
    raw = np.zeros(20)  # pathological: every user's raw risk is exactly 0
    post = np.zeros(20)
    out = decile_reduction_ci(raw, post, deciles, n_boot=200)
    assert out[0]["relative_reduction"]["n_boot_valid"] == 0  # all raw_mean==0 draws excluded, not NaN-propagated
    assert out[0]["absolute_reduction"]["mean"] == 0.0
    print("OK: all-zero raw risk degrades gracefully (relative reduction empty, absolute reduction 0) "
          "instead of crashing or returning garbage")


if __name__ == "__main__":
    test_absolute_reduction_matches_manual_decile_means()
    test_relative_reduction_handles_known_ratio()
    test_residual_risk_disparity_known_ratio()
    test_near_zero_raw_risk_does_not_crash_and_is_excluded_from_relative()
    print("All decile_reduction tests passed.")
