"""Phase 3.3 leveling-down check, RECALIBRATED per the PI (2026-10-27).
Two changes from the original version:

1. Uses absolute post-mechanism risk directly (delta(u) = post_baseline(u)
   - post_mechanism(u)), not the per-user ratio `privacy_gain`. Raw risk
   cancels out of this difference entirely (both terms are
   raw(u) - post_x(u), so the per-user ratio's instability near
   raw(u) ~ 0 -- flagged throughout this project as unstable -- cannot
   enter here at all.
2. "Harmed"/"helped" are no longer a fixed epsilon (+-1e-9) around zero.
   They are derived from a SEED-TO-SEED NULL: the same baseline
   mechanism's own encoder-seed-to-encoder-seed variation (e.g. plain
   Mo-PAE's post_top1_encseed0 vs. post_top1_encseed1 vs. ...encseed2),
   which is exactly the comparison the PI asked for ("plain vs plain
   across encoder seeds") and directly measures how much apparent
   harm/help would appear from encoder-training noise alone, including
   regression to the mean (a decile's already-extreme users look less
   extreme on a repeat measurement with no real effect). A delta only
   counts as real harm/help if it falls outside that null's own
   [alpha/2, 1-alpha/2] quantiles, and the observed harmed-rate is
   bootstrap-tested against the null's own expected harmed-rate before
   calling anything "leveling down."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seed_to_seed_null(post_seed_cols: list) -> np.ndarray:
    """Builds the null distribution of post-risk DIFFERENCES from a single
    mechanism's own encoder-seed replicates (e.g. [df['post_top1_encseed0'],
    df['post_top1_encseed1'], df['post_top1_encseed2']] for a single plain-
    Mo-PAE run, same users throughout, only the encoder's random
    initialization differs). Returns all ordered pairwise differences,
    pooled across all seed pairs and all users -- this is the null a real
    mechanism-vs-baseline delta is compared against. Caller is responsible
    for restricting `post_seed_cols` to one decile's users if a per-decile
    null is wanted (it should be -- noise characteristics vary by decile)."""
    diffs = []
    n = len(post_seed_cols)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diffs.append((np.asarray(post_seed_cols[i]) - np.asarray(post_seed_cols[j])))
    return np.concatenate(diffs) if diffs else np.array([])


def leveling_down_check(post_mechanism: pd.Series, post_baseline: pd.Series,
                         null_deltas: np.ndarray, n_boot: int = 2000, seed: int = 0,
                         ci: float = 0.95) -> dict:
    """post_mechanism, post_baseline indexed by a common key (caller's
    responsibility that both cover the same users) -- both are POST-
    MECHANISM RISK (e.g. post_top1), not gain. delta(u) = post_baseline(u)
    - post_mechanism(u): positive means the mechanism's post-risk is LOWER
    than baseline's for this user (helped, more protected under the
    mechanism); negative means HIGHER (harmed, less protected).
    null_deltas: the seed-to-seed null (see seed_to_seed_null) from the
    SAME baseline mechanism's own encoder-seed replicates, for the SAME
    decile's users.

    Returns frac_harmed/helped/unchanged using null-derived thresholds (not
    a fixed epsilon), plus a bootstrap test of whether the observed harmed
    rate exceeds what the null itself would produce, and -- only if that
    test is affirmative -- whether harmed users were disproportionately
    the ones the baseline already protected well (the original leveling-
    down pattern, restated in risk terms: harmed users had LOWER baseline
    risk than the group average, i.e. were already better-protected)."""
    post_mechanism, post_baseline = post_mechanism.align(post_baseline, join="inner")
    delta = post_baseline - post_mechanism
    alpha = (1 - ci) / 2

    out = dict(n_users=int(len(delta)), mean_delta=float(delta.mean()) if len(delta) else float("nan"))
    if len(null_deltas) < 10:
        out["error"] = "null too small (<10 points) to derive thresholds"
        return out
    if len(delta) < 2:
        out["error"] = "too few users to classify"
        return out

    lo_thresh = float(np.quantile(null_deltas, alpha))
    hi_thresh = float(np.quantile(null_deltas, 1 - alpha))
    out["null_lo_thresh"], out["null_hi_thresh"] = lo_thresh, hi_thresh
    out["null_n"] = int(len(null_deltas))

    harmed = delta < lo_thresh
    helped = delta > hi_thresh
    unchanged = ~harmed & ~helped
    out.update(
        frac_harmed=float(harmed.mean()), frac_helped=float(helped.mean()),
        frac_unchanged=float(unchanged.mean()),
    )

    rng = np.random.default_rng(seed)
    null_harmed_rate_boot = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.choice(null_deltas, size=len(delta), replace=True)
        null_harmed_rate_boot[i] = (draw < lo_thresh).mean()
    out["null_expected_harmed_rate"] = float(null_harmed_rate_boot.mean())
    out["null_harmed_rate_ci_hi"] = float(np.quantile(null_harmed_rate_boot, 1 - alpha))
    out["harmed_rate_exceeds_null"] = bool(harmed.mean() > out["null_harmed_rate_ci_hi"])

    if out["harmed_rate_exceeds_null"] and harmed.sum() >= 2:
        harmed_baseline_risk = float(post_baseline[harmed].mean())
        all_baseline_risk = float(post_baseline.mean())
        out["harmed_users_baseline_risk_mean"] = harmed_baseline_risk
        out["all_users_baseline_risk_mean"] = all_baseline_risk
        out["leveling_down_pattern"] = harmed_baseline_risk < all_baseline_risk
    else:
        out["leveling_down_pattern"] = False
        out["leveling_down_reason"] = ("harmed rate not distinguishable from the seed-to-seed null"
                                        if not out["harmed_rate_exceeds_null"] else "fewer than 2 harmed users")
    return out
