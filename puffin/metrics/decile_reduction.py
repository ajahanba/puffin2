"""Replaces the per-user-ratio `privacy_gain` in summary tables, per the
PI: that metric averages a RATIO per user (1 - post(u)/raw(u)), which is
unstable whenever a user's raw risk is near zero (a recurring hazard
flagged throughout this project -- e.g. Checkpoint 3b's decile 4 case:
raw=0.001, post=0.077 -> gain=-76). This module instead works from
DECILE-LEVEL MEANS: absolute reduction = mean(raw) - mean(post), relative
reduction = 1 - mean(post)/mean(raw) -- both computed on aggregated,
not per-user, quantities, with bootstrap CIs that jointly resample raw
and post together (the same resampled users feed both numbers each
draw) so the reduction reflects genuine within-sample pairing.

The old per-user ratio (`puffin.metrics.privacy.privacy_gain`) is kept
unchanged and is still reported, but only in an appendix, labeled
unstable -- this module does not replace it, callers choose which to use
for the headline table.
"""
from __future__ import annotations

import numpy as np


def decile_reduction_ci(raw: np.ndarray, post: np.ndarray, deciles: np.ndarray,
                         n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Per decile: raw_mean, post_mean, absolute reduction (raw_mean -
    post_mean) and relative reduction (1 - post_mean/raw_mean), each with
    a bootstrap 95% CI from jointly resampling that decile's users (same
    draw feeds both raw and post each iteration, not two independent
    resamples)."""
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    out = {}
    for d in sorted(np.unique(deciles)):
        mask = deciles == d
        r, p = raw[mask], post[mask]
        n = len(r)
        if n == 0:
            continue
        abs_boot = np.empty(n_boot)
        rel_boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            rm, pm = r[idx].mean(), p[idx].mean()
            abs_boot[i] = rm - pm
            rel_boot[i] = (1 - pm / rm) if rm > 0 else np.nan
        rel_boot = rel_boot[~np.isnan(rel_boot)]
        out[int(d)] = dict(
            n_users=int(n), raw_mean=float(r.mean()), post_mean=float(p.mean()),
            absolute_reduction=dict(mean=float(abs_boot.mean()),
                                     ci_lo=float(np.quantile(abs_boot, alpha)),
                                     ci_hi=float(np.quantile(abs_boot, 1 - alpha)),
                                     ci_excludes_zero=bool(np.quantile(abs_boot, alpha) > 0
                                                            or np.quantile(abs_boot, 1 - alpha) < 0)),
            relative_reduction=dict(mean=float(rel_boot.mean()) if len(rel_boot) else float("nan"),
                                     ci_lo=float(np.quantile(rel_boot, alpha)) if len(rel_boot) else float("nan"),
                                     ci_hi=float(np.quantile(rel_boot, 1 - alpha)) if len(rel_boot) else float("nan"),
                                     n_boot_valid=int(len(rel_boot))),
        )
    return out


def residual_risk_disparity(post: np.ndarray, deciles: np.ndarray, low_decile: int = 0,
                             high_decile: int = 9, n_boot: int = 2000, seed: int = 0,
                             ci: float = 0.95) -> dict:
    """Ratio of post-mechanism risk between the lowest- and highest-
    entropy deciles (post_low / post_high), with a bootstrap CI: each
    iteration independently resamples users WITHIN each decile group
    (with replacement, same group sizes), computes each group's mean,
    and takes the ratio -- the standard way to get a valid CI on a
    ratio-of-two-group-means statistic."""
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    low = post[deciles == low_decile]
    high = post[deciles == high_decile]
    if len(low) == 0 or len(high) == 0:
        return dict(error=f"missing decile {low_decile} or {high_decile}",
                    n_low=int(len(low)), n_high=int(len(high)))
    ratio_boot = np.empty(n_boot)
    for i in range(n_boot):
        lm = low[rng.integers(0, len(low), size=len(low))].mean()
        hm = high[rng.integers(0, len(high), size=len(high))].mean()
        ratio_boot[i] = lm / hm if hm > 0 else np.nan
    ratio_boot = ratio_boot[~np.isnan(ratio_boot)]
    point = float(low.mean() / high.mean()) if high.mean() > 0 else float("nan")
    return dict(
        low_decile=low_decile, high_decile=high_decile, n_low=int(len(low)), n_high=int(len(high)),
        point_estimate=point,
        mean=float(ratio_boot.mean()) if len(ratio_boot) else float("nan"),
        ci_lo=float(np.quantile(ratio_boot, alpha)) if len(ratio_boot) else float("nan"),
        ci_hi=float(np.quantile(ratio_boot, 1 - alpha)) if len(ratio_boot) else float("nan"),
        n_boot_valid=int(len(ratio_boot)),
    )
