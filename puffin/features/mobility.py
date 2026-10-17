"""Phase 2 task 1: per-user mobility features for the disparate-risk motivation.

Three entropies, following Song, Qu, Blumm & Barabasi (2010), "Limits of
Predictability in Human Mobility", Science 327(5968):
  - S_rand: random entropy, log2(number of distinct locations).
  - S_unc: uncorrelated (temporal) entropy, Shannon entropy of the
    visitation-frequency distribution, ignoring order. (Same quantity as
    puffin.data.stats.per_user_entropy -- recomputed here, not imported, so
    this module has no dependency on the Phase 1 stats module and can be
    applied to any dataframe with user_id/cell_id.)
  - S_real: the "actual" entropy rate, estimated via the Lempel-Ziv
    estimator (Kontoyiannis et al. 1998; the same estimator Song et al.
    use), which captures the sequential/temporal structure the other two
    ignore.

Plus radius of gyration (Gonzalez, Hidalgo & Barabasi 2008) and "regularity"
(share of all visits falling on the user's own top-2 most-visited cells, the
plan's specific definition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

# S_real's Lempel-Ziv estimator is at best O(n * match_length) per user; on a
# raw GPS trace resampled every few seconds, consecutive repeats at the
# sampling rate would swamp it with trivial self-matches and wouldn't be
# measuring "real" mobility entropy at all -- the sequence is first collapsed
# (consecutive repeated cell_id -> one visit) and then capped, since GeoLife's
# longest user trace is 2.1M raw points. Both are logged in the output so
# callers can see exactly how much of each user's history fed into S_real.
MAX_VISITS_FOR_S_REAL = 3000


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _collapse_consecutive(seq: np.ndarray) -> np.ndarray:
    """Collapse runs of consecutive identical values to a single value."""
    if len(seq) == 0:
        return seq
    keep = np.ones(len(seq), dtype=bool)
    keep[1:] = seq[1:] != seq[:-1]
    return seq[keep]


def _lz_entropy_rate(seq: list) -> float | None:
    """Kontoyiannis et al. (1998) / Song et al. (2010) Lempel-Ziv entropy-rate
    estimator: S_real = [ (1/n) sum_i Lambda_i ]^-1 * log2(n), where Lambda_i
    is 1 + the length of the longest substring starting at i that has
    occurred somewhere in positions [0, i-1]. Returns None if n is too small
    for the estimator to mean anything (n < 2).

    Implemented via Python string search (`str.find`, C-optimized) rather
    than a manual nested Python loop over candidate start positions -- the
    latter is correct but, at MAX_VISITS_FOR_S_REAL=3000, was measured to be
    far too slow (effectively O(n^2) of pure-Python-level comparisons) to
    run across every user. Each location id maps to one Unicode code point
    (cell ids are always >=1 and max out at GeoLife's 85,074, far below the
    0x10FFFF limit), so a sequence becomes a short string and substring
    search is delegated to CPython's C implementation."""
    n = len(seq)
    if n < 2:
        return None
    s = "".join(chr(c) for c in seq)
    lambdas = []
    for i in range(n):
        if i == 0:
            lambdas.append(1)
            continue
        prefix_pool = s[:i]
        L = 1
        while i + L <= n and prefix_pool.find(s[i:i + L]) != -1:
            L += 1
        lambdas.append(L)
    mean_lambda = float(np.mean(lambdas))
    if mean_lambda <= 0:
        return None
    return float(np.log2(n) / mean_lambda)


def compute_mobility_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have user_id, cell_id, lat, lon, ts (sortable). Returns one
    row per user_id with s_rand, s_unc, s_real, s_real_n_visits_used (how
    many collapsed visits actually fed S_real, capped at
    MAX_VISITS_FOR_S_REAL), rog_km (radius of gyration), regularity_top2,
    n_distinct_cells, n_records."""
    rows = []
    for user_id, g in df.sort_values("ts").groupby("user_id"):
        cells = g["cell_id"].to_numpy()
        n_records = len(cells)
        n_distinct = len(np.unique(cells))

        s_rand = float(np.log2(n_distinct)) if n_distinct > 0 else 0.0

        counts = pd.Series(cells).value_counts()
        probs = counts / counts.sum()
        s_unc = float(-np.sum(probs * np.log2(probs)))

        collapsed = _collapse_consecutive(cells)
        visits_for_lz = [int(c) for c in collapsed[:MAX_VISITS_FOR_S_REAL]]
        s_real = _lz_entropy_rate(visits_for_lz)

        lat_cm, lon_cm = g["lat"].mean(), g["lon"].mean()
        dists = _haversine_km(g["lat"].to_numpy(), g["lon"].to_numpy(), lat_cm, lon_cm)
        rog_km = float(np.sqrt(np.mean(dists ** 2)))

        top2_share = float(counts.sort_values(ascending=False).head(2).sum() / counts.sum())

        rows.append(dict(
            user_id=user_id, n_records=n_records, n_distinct_cells=n_distinct,
            s_rand=s_rand, s_unc=s_unc, s_real=s_real,
            s_real_n_visits_used=len(visits_for_lz),
            rog_km=rog_km, regularity_top2=top2_share,
        ))
    return pd.DataFrame(rows)
