"""Phase 5: GI-DP baseline -- Geo-Indistinguishability location perturbation.

Per `puffin/attacks/README.md`'s threat-model table, GI-DP releases
PERTURBED (noised) location sequences (not encoder features like Mo-PAE/
PUFFIN), so it is attacked with TUL (`puffin/attacks/tul_maintul.py`) and
re-id directly on the perturbed sequences (`puffin/attacks/reid.py`), never
the feature-based attacks. This module builds the mechanism itself; running
the attack suite against it is a later phase's job (per the task's own
scope).

## What "GI-DP" means here, and why this is an in-house build, not a port

Unlike TrajGAN, "GI-DP" is not a specific repo/paper name -- it is read
here as shorthand for **Geo-Indistinguishability (Andres, Bordenabe,
Chatzikokolakis & Palamidessi, CCS 2013, arXiv:1212.1984)**, the classical
differential-privacy-flavored location-perturbation mechanism, per the
task's own framing. A genuine search was made for an existing open-source
implementation before deciding to build in-house:

- **GeoPrivacy** (PyPI, `pip install GeoPrivacy`) implements the Planar
  Laplace mechanism's noise sampler (`random_laplace_noise`), but its own
  README frames it as covering "two privacy mechanisms for the **sporadic
  case**" -- single independent location reports, with its own worked
  example being one NYC taxi pickup, not a trajectory. It also has no
  schema/dataframe integration of any kind -- adopting it would mean
  depending on a tiny single-function PyPI package for exactly the
  textbook formula (see below) and still writing all of the trajectory-
  level integration (schema I/O, re-tessellation, batch application)
  ourselves.
- **privkit** (`privkit.fc.up.pt`) implements an *Adaptive* Geo-
  Indistinguishability variant (Al-Dhubhani & Cazalas 2018) that adjusts
  epsilon per-point based on correlation with past locations -- a
  different, more elaborate mechanism than "the classical mechanism" the
  task asks for, and pulling in a whole external privacy-metrics
  framework to get one baseline formula would be a worse trade than
  writing the ~15 lines of actual math directly.
- No repository literally named "GI-DP" implementing a trajectory-level
  mechanism was found on GitHub or via web search.

Given that, and given the task's own explicit license to use judgment
here, this baseline is a clean in-house implementation of the **Planar
Laplace mechanism** exactly as specified in the original 2013 paper
(Theorem 3 / Section 4.2) -- the paper gives a complete, unambiguous,
three-line algorithm (sample an angle uniformly, sample a radius from a
closed-form inverse CDF, convert to a location offset), so reimplementing
it directly from the formula is both faster and more verifiable than
reading a third-party implementation whose provenance would need equal
scrutiny. Same category of decision as this project's `mia_inhouse.py`
(POIPrivacy ruled out, replaced by a textbook Yeom/Shokri MIA implemented
directly from its own well-known formula), except here there wasn't a
genuinely-blocking repo to rule out -- rather, no fitting repo was found at
all after a real search, which is what led to the in-house path here.

## The mechanism

For a true location (lat, lon), draw:

  theta ~ Uniform(0, 2*pi)                         -- noise direction
  p     ~ Uniform(0, 1)
  r     = -(1/epsilon) * (W_{-1}((p-1)/e) + 1)      -- noise radius (meters)

where `W_{-1}` is the lower (real, non-principal) branch of the Lambert W
function. This is the exact inverse-CDF sampler for the Planar Laplace
distribution's radius, derived in the paper from its CDF
`C_eps(r) = 1 - (1 + eps*r) * exp(-eps*r)`: setting `p = C_eps(r)` and
solving for `r` gives the formula above (verified against the paper's own
Section 4.2 and cross-checked against the independently-derived GeoPrivacy
package's README, which states the same CDF). `epsilon` has units of
1/meter (the paper's own convention: smaller epsilon -> more noise -> more
privacy, exactly like standard differential privacy's epsilon, just scaled
per unit distance instead of per dataset). The offset (r, theta) is
converted from meters to a (lat, lon) degree offset via the standard local
equirectangular approximation (`METERS_PER_DEGREE_LAT`, with the
longitude term scaled by `cos(latitude)` since a degree of longitude
covers less ground away from the equator) -- the same conversion used by
essentially every GI implementation that has to work in lat/lon rather
than a flat projected plane (including the Location Guard browser
extension, a widely-cited deployed use of this exact mechanism).

## Scope: a location mechanism, applied per-point to a trajectory

This is the base, single-location mechanism exactly as published -- it has
no notion of a trajectory at all. Applying it independently to every point
of a real trajectory (what `run_gi_dp` below does, since this project's
unit of data is always a trajectory) means the noise draws are
independent across points: the whole trajectory is NOT
epsilon-geo-indistinguishable as a unit, only each individual point is (in
isolation). This is a well-known, deliberate limitation of the base
mechanism, not a bug in this port -- addressing it is exactly what later
"budget-splitting" / temporal-correlation-aware GI variants
(e.g. Al-Dhubhani & Cazalas 2018, the privkit mechanism ruled out above)
exist for, and is out of scope for "the classical mechanism," which is
what the task asked for. Documented here rather than silently assumed.

`weekday`/`hour` are copied through unperturbed -- the published mechanism
has no temporal component; only location is privatized. `cell_id` is
re-tessellated from the perturbed (lat, lon) via
`puffin.data.tessellation.assign_grid_cells` (same function Phase 1 used
to build every real dataset's `cell_id`), for the same reason
`trajgan.py` does this: downstream attacks (`tul_maintul.py`, `reid.py`)
consume `cell_id`, and the mechanism's own output is continuous lat/lon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import lambertw

from puffin.data.tessellation import assign_grid_cells

METERS_PER_DEGREE_LAT = 111_320.0


def _sample_radius(epsilon: float, p: np.ndarray) -> np.ndarray:
    """Inverse-CDF sample of the Planar Laplace radius for CDF value(s)
    `p` (see module docstring for the derivation). `p` must be in [0, 1);
    values very close to 1 (vanishing probability, 1-in-10^dozens at
    typical sample sizes) can numerically push `lambertw` to -inf -- the
    caller draws `p` from `Uniform(0,1)` so this is a true-zero-probability
    edge case in practice, not something normal use ever hits."""
    x = (p - 1.0) / np.e
    w = lambertw(x, k=-1).real
    return -(1.0 / epsilon) * (w + 1.0)


def planar_laplace_noise(epsilon: float, n: int, rng: np.random.Generator):
    """Draws `n` independent Planar Laplace noise vectors in METERS,
    returned as (dx, dy) -- `dx` eastward, `dy` northward. Theoretical
    E[r] = 2/epsilon (a standard closed-form property of this
    distribution); `tests/test_gi_dp.py` checks the empirical mean
    converges to this."""
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    p = rng.uniform(0.0, 1.0, size=n)
    r = _sample_radius(epsilon, p)
    return r * np.cos(theta), r * np.sin(theta)


def perturb_latlon(lat: np.ndarray, lon: np.ndarray, epsilon: float, rng: np.random.Generator):
    """Applies the Planar Laplace mechanism independently to each (lat,
    lon) pair. Returns (perturbed_lat, perturbed_lon), same shape as the
    inputs."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dx_m, dy_m = planar_laplace_noise(epsilon, len(lat), rng)
    dlat = dy_m / METERS_PER_DEGREE_LAT
    dlon = dx_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def run_gi_dp(df: pd.DataFrame, epsilon: float = 0.01, seed: int = 0,
              cell_mapping: dict | None = None) -> pd.DataFrame:
    """Applies Planar Laplace location perturbation to every row of `df`
    (every point treated as an independent release -- see module
    docstring's "Scope" section). `epsilon` is in units of 1/meter;
    smaller = more noise = more private (standard convention; e.g.
    epsilon=0.01 means ~2/0.01=200m mean perturbation radius).

    `cell_mapping`: pass the ORIGINAL (unperturbed) dataset's grid-cell
    mapping (`df.attrs["grid_cell_mapping"]` after calling
    `assign_grid_cells` on it, or build one via
    `assign_grid_cells(original_df, resolution_deg=0.01).attrs[
    "grid_cell_mapping"]`) so the perturbed release's cell_id stays in the
    SAME numbering as the original -- without this, re-tessellating the
    perturbed coordinates builds an independent dense mapping, and the
    resulting cell_id values are not comparable to the original's at all
    (same physical cell gets a different integer); see
    puffin.data.tessellation.assign_grid_cells's docstring for the bug
    this was found fixing (2026-11-04). Any physical cell not in
    `cell_mapping` (a point perturbed outside the original's observed
    footprint) gets a newly-assigned id extending that mapping, same as
    assign_grid_cells's own extension behavior.

    Returns a unified-schema dataframe (`puffin.data.schema.
    REQUIRED_COLUMNS`): perturbed `lat`/`lon`, re-tessellated `cell_id`,
    `user_id`/`traj_id`/`ts`/`weekday`/`hour` copied through unchanged
    (the mechanism only touches location)."""
    required = {"user_id", "traj_id", "ts", "lat", "lon", "weekday", "hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"run_gi_dp: missing required columns: {missing}")

    rng = np.random.default_rng(seed)
    carry_through = ["user_id", "traj_id", "ts", "weekday", "hour"]
    if "split" in df.columns:
        carry_through.append("split")  # perturbation touches location only; train/test membership is unchanged
    out = df[carry_through].copy()
    out["lat"], out["lon"] = perturb_latlon(df["lat"].to_numpy(), df["lon"].to_numpy(), epsilon, rng)
    out = assign_grid_cells(out, resolution_deg=0.01, existing_mapping=cell_mapping)
    return out[carry_through + ["lat", "lon", "cell_id"]]
