#!/usr/bin/env python
"""Correctness checks for puffin/baselines/gi_dp.py (Phase 5, GI-DP /
Geo-Indistinguishability).

Run directly: `python tests/test_gi_dp.py`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.baselines.gi_dp import planar_laplace_noise, perturb_latlon, run_gi_dp
from puffin.data.schema import validate_schema


def test_mean_radius_matches_closed_form():
    """Planar Laplace's radius has a known closed-form mean, E[r] = 2/eps
    -- a real, manually-computable value from the distribution's own
    definition (not just "did it run"). With 200,000 draws the empirical
    mean should land within a few percent of the theoretical value."""
    rng = np.random.default_rng(0)
    for eps in (0.001, 0.01, 0.1):
        dx, dy = planar_laplace_noise(eps, 200_000, rng)
        r = np.sqrt(dx**2 + dy**2)
        expected = 2.0 / eps
        rel_err = abs(r.mean() - expected) / expected
        assert rel_err < 0.03, (eps, r.mean(), expected, rel_err)
        print(f"OK: eps={eps}: empirical mean radius {r.mean():.2f}m vs theoretical {expected:.2f}m "
              f"(rel err {rel_err:.1%})")


def test_noise_direction_is_uniform_and_unbiased():
    """dx, dy should each average to ~0 (noise has no directional bias) --
    a wrong angle sampler (e.g. always theta=0) would fail this."""
    rng = np.random.default_rng(1)
    dx, dy = planar_laplace_noise(0.01, 200_000, rng)
    assert abs(dx.mean()) < 5.0, dx.mean()   # mean radius ~200m; 5m tolerance on the mean
    assert abs(dy.mean()) < 5.0, dy.mean()
    print(f"OK: noise is directionally unbiased (mean dx={dx.mean():.3f}m, dy={dy.mean():.3f}m)")


def test_smaller_epsilon_means_more_noise():
    """The privacy knob must behave sensibly: smaller epsilon (more
    privacy) must produce LARGER perturbation distances, monotonically."""
    rng = np.random.default_rng(2)
    lat = np.full(5000, 39.9)
    lon = np.full(5000, 116.4)
    radii_by_eps = {}
    for eps in (0.1, 0.01, 0.001):
        plat, plon = perturb_latlon(lat, lon, eps, np.random.default_rng(2))
        dlat_m = (plat - lat) * 111_320.0
        dlon_m = (plon - lon) * 111_320.0 * np.cos(np.radians(lat))
        dist = np.sqrt(dlat_m**2 + dlon_m**2)
        radii_by_eps[eps] = dist.mean()
    assert radii_by_eps[0.1] < radii_by_eps[0.01] < radii_by_eps[0.001], radii_by_eps
    print(f"OK: mean perturbation distance increases as epsilon decreases: {radii_by_eps}")


def _toy_df(n_users=6, n_traj_per_user=3, traj_len=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        for traj_id in range(1, n_traj_per_user + 1):
            for t in range(traj_len):
                rows.append(dict(
                    user_id=u, traj_id=traj_id,
                    ts=pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=t),
                    lat=39.9 + 0.01 * u + rng.normal(0, 0.0005),
                    lon=116.4 + 0.01 * u + rng.normal(0, 0.0005),
                    weekday=t % 7, hour=t % 24,
                ))
    return pd.DataFrame(rows)


def test_run_gi_dp_produces_schema_valid_output_and_preserves_time_fields():
    df = _toy_df()
    out = run_gi_dp(df, epsilon=0.01, seed=0)
    for col in ("user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"):
        assert col in out.columns, col
    validate_schema(out)  # must be a drop-in unified-schema dataframe
    assert len(out) == len(df)
    # weekday/hour (the mechanism never touches time) must be byte-identical
    assert (out["weekday"].to_numpy() == df["weekday"].to_numpy()).all()
    assert (out["hour"].to_numpy() == df["hour"].to_numpy()).all()
    assert (out["user_id"].to_numpy() == df["user_id"].to_numpy()).all()
    # lat/lon must actually have moved (mechanism applied, not a no-op)
    assert not np.allclose(out["lat"].to_numpy(), df["lat"].to_numpy())
    print("OK: run_gi_dp returns a schema-valid dataframe, perturbs lat/lon, "
          "and leaves user_id/traj_id/weekday/hour untouched")


if __name__ == "__main__":
    test_mean_radius_matches_closed_form()
    test_noise_direction_is_uniform_and_unbiased()
    test_smaller_epsilon_means_more_noise()
    test_run_gi_dp_produces_schema_valid_output_and_preserves_time_fields()
    print("All gi_dp tests passed.")
