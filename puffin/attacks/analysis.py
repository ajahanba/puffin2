"""Bootstrap confidence intervals for per-decile and paired risk statistics."""
from __future__ import annotations

import numpy as np


def bootstrap_decile_ci(values: np.ndarray, deciles: np.ndarray, n_boot: int = 2000,
                         seed: int = 0, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    out = {}
    for d in sorted(np.unique(deciles)):
        vals = values[deciles == d]
        n = len(vals)
        if n == 0:
            continue
        boots = np.array([vals[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
        ci_lo, ci_hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
        out[int(d)] = dict(mean=float(vals.mean()), ci_lo=ci_lo, ci_hi=ci_hi, n_users=n,
                            ci_excludes_zero=bool(ci_lo > 0 or ci_hi < 0))
    return out


def paired_bootstrap_test(values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 2000,
                           seed: int = 0, ci: float = 0.95) -> dict:
    """Paired test between two methods on the same users: a and b must be
    the same length, same user order (paired by construction, e.g. two
    methods' privacy_gain for the same user_id). Bootstraps the mean
    difference (a - b) over users; reports whether its CI excludes zero."""
    assert len(values_a) == len(values_b), "paired test needs equal-length, aligned arrays"
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    ci_lo, ci_hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
    return dict(mean_diff=float(diff.mean()), ci_lo=ci_lo, ci_hi=ci_hi, n=n,
                ci_excludes_zero=bool(ci_lo > 0 or ci_hi < 0))
