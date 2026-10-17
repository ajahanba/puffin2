#!/usr/bin/env python
"""Correctness + performance sanity check for puffin/features/mobility.py."""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features, _lz_entropy_rate, _collapse_consecutive


def test_collapse_consecutive():
    out = _collapse_consecutive(np.array([1, 1, 1, 2, 2, 3, 1, 1]))
    assert list(out) == [1, 2, 3, 1], list(out)
    print("OK: _collapse_consecutive")


def test_lz_entropy_rate_known_cases():
    # A perfectly periodic sequence is maximally predictable -> low entropy rate.
    periodic = [1, 2, 3] * 200
    # A fully random sequence over a large alphabet is unpredictable -> higher entropy rate.
    rng = np.random.default_rng(0)
    random_seq = list(rng.integers(0, 1000, size=600))

    s_periodic = _lz_entropy_rate(periodic)
    s_random = _lz_entropy_rate(random_seq)
    assert s_periodic < s_random, (s_periodic, s_random)
    print(f"OK: periodic S_real={s_periodic:.3f} < random S_real={s_random:.3f}")


def test_lz_entropy_rate_performance():
    rng = np.random.default_rng(0)
    seq = list(rng.integers(0, 500, size=3000))
    t0 = time.time()
    _lz_entropy_rate(seq)
    dt = time.time() - t0
    print(f"OK: LZ estimator on n=3000 took {dt:.2f}s")
    assert dt < 30, f"too slow: {dt:.2f}s for n=3000 (must be tractable for ~180 users)"


def test_compute_mobility_features_runs():
    df = pd.DataFrame({
        "user_id": [1] * 10 + [2] * 10,
        "cell_id": [1, 1, 2, 2, 3, 1, 1, 2, 3, 3] + [5, 6, 7, 8, 5, 6, 7, 8, 5, 6],
        "lat": np.linspace(40.0, 40.01, 20),
        "lon": np.linspace(-73.0, -73.01, 20),
        "ts": list(range(10)) + list(range(10)),
    })
    out = compute_mobility_features(df)
    assert set(out["user_id"]) == {1, 2}
    assert (out["n_records"] == 10).all()
    print("OK: compute_mobility_features runs and returns one row per user")
    print(out)


if __name__ == "__main__":
    test_collapse_consecutive()
    test_lz_entropy_rate_known_cases()
    test_lz_entropy_rate_performance()
    test_compute_mobility_features_runs()
    print("All mobility feature tests passed.")
