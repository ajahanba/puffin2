"""Phase 3.3 / Phase 4 individual-fairness metrics over per-user privacy
gain (puffin.metrics.privacy.privacy_gain). All take `gain`: (n_users,)
array, indexed consistently with any `distance_matrix`: (n_users, n_users)
from puffin.fairness.similarity (Phase 4) or an ad hoc distance for now.
"""
from __future__ import annotations

import numpy as np


def similar_pairs_within_epsilon(gain: np.ndarray, distance_matrix: np.ndarray,
                                  epsilon: float, delta: float) -> float:
    """The original PUFFIN metric, kept for continuity: among pairs (i, j)
    considered "similar" (d(i,j) < delta), what fraction have gain within
    epsilon of each other? NaN if no pairs are close enough to count."""
    n = len(gain)
    iu = np.triu_indices(n, k=1)
    close = distance_matrix[iu] < delta
    if not close.any():
        return float("nan")
    gain_diff = np.abs(gain[:, None] - gain[None, :])[iu]
    return float((gain_diff[close] <= epsilon).mean())


def lipschitz_violation(gain: np.ndarray, distance_matrix: np.ndarray, L: float) -> dict:
    """Phase 4's individual-fairness notion: |gain_i - gain_j| should be
    <= L * d(i,j) for every pair. Reports the fraction of pairs that
    violate this and the mean size of the violation (0 if none)."""
    n = len(gain)
    iu = np.triu_indices(n, k=1)
    gain_diff = np.abs(gain[:, None] - gain[None, :])[iu]
    bound = (L * distance_matrix)[iu]
    violation = np.clip(gain_diff - bound, 0, None)
    has_violation = violation > 0
    return dict(
        violation_rate=float(has_violation.mean()),
        mean_violation_size=float(violation[has_violation].mean()) if has_violation.any() else 0.0,
        n_pairs=int(len(violation)),
    )


def max_min_gain(gain: np.ndarray) -> float:
    """Rawlsian worst-off-user's gain."""
    return float(np.min(gain))


def gini(gain: np.ndarray) -> float:
    """Gini coefficient of the gain distribution. Gain can be negative
    (harm); the standard Gini formula assumes non-negative values, so
    negative gains are shifted up by the minimum before computing --
    documented here because it changes the number's meaning (it becomes
    "inequality of gain relative to the worst-off user," not "inequality
    of gain relative to zero") and should be stated wherever this is
    reported, not left implicit."""
    g = np.sort(np.asarray(gain, dtype=float))
    if g.min() < 0:
        g = g - g.min()
    n = len(g)
    cum = np.cumsum(g)
    if cum[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def gain_percentiles(gain: np.ndarray, percentiles=(10, 50, 90)) -> dict:
    return {int(p): float(np.percentile(gain, p)) for p in percentiles}
