#!/usr/bin/env python
"""Item 2 REDO (PI correction, 2026-11-09): the original compare() drew
an INDEPENDENT user-resample per curve point instead of reusing the
mechanism's draw, breaking the pairing -- confirmed the exact bug the
PI found by hand (GeoLife decile0 should be [-0.20,-0.06], not
[-0.313,+0.058]).

Corrected design, per the PI's (a)-(d):
(a) one user index vector per replicate, applied to both arms;
(b) decile membership fixed from the CSV's entropy_decile, read once;
(c) encoder seeds resampled within each run, SAME seed draw used for
    both lambda3 endpoints of the interpolation (not independently
    per curve point);
(d) difference computed per user, then aggregated.

Also: the interpolation weight t is now computed ONCE from the point-
estimate (seed-averaged) utilities of the mechanism and the two
bracketing curve points -- not re-derived from noisy resampled utility
every replicate (that was never requested and adds needless noise).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm

MIN_WINDOWS = 10
N_BOOT = 2000
TOP_FRAC = 0.10
SIM_BASES = ("entropy", "location_distribution", "trajectory_shape", "radius_of_gyration")


def load_point(run_dir):
    import json
    summ = json.load(open(run_dir / "summary.json"))
    df = pd.read_csv(run_dir / "per_user_04d.csv")
    df = df[df.n_windows >= MIN_WINDOWS].copy()
    seed_cols = sorted([c for c in df.columns if c.startswith("post_top1_encseed")],
                        key=lambda c: int(c.replace("post_top1_encseed", "")))
    util = np.array(summ["utility_per_encoder_seed"])
    return dict(util=util, n_seeds=len(seed_cols), df=df.set_index("user_id"),
                seed_cols=seed_cols, source=run_dir.name)


def find_bracket(mech_util_point, curve_points, lambda3_values):
    """Returns (lo, hi, lo_util, hi_util) bracketing mech_util_point by
    walking the lambda3 SWEEP in its own ascending order (not sorting
    curve points by their realized utility) and finding the first
    adjacent pair where utility crosses mech_util_point. The curve is
    not perfectly monotonic (GeoLife's lambda3 in [0.02,0.8] is a near-
    flat, noisy region where utility doesn't move much), so sorting by
    utility can pick a far-away, same-utility-level point instead of
    the methodologically correct adjacent-in-lambda3 bracket -- PI-
    caught, confirmed against the hand-derived lambda3=0.8/1.2 case."""
    order = np.argsort(lambda3_values)
    sorted_pts = [curve_points[i] for i in order]
    sorted_lp = [lambda3_values[i] for i in order]
    sorted_utils = [cp["util"].mean() for cp in sorted_pts]
    for i in range(len(sorted_utils) - 1):
        u_a, u_b = sorted_utils[i], sorted_utils[i + 1]
        lo_u, hi_u = min(u_a, u_b), max(u_a, u_b)
        if lo_u <= mech_util_point <= hi_u:
            # orient so the FIRST returned point is the lower-utility one
            if u_a <= u_b:
                return sorted_pts[i], sorted_pts[i + 1], u_a, u_b
            return sorted_pts[i + 1], sorted_pts[i], u_b, u_a
    return None


def align_risk(point, users):
    return point["df"].loc[users, point["seed_cols"]].to_numpy().T  # (n_seeds, n_users)


def paired_bootstrap_matched(mech_point, lo_point, hi_point, users, rng,
                              n_boot=N_BOOT, resample_seeds=True):
    """Corrected paired bootstrap: PUFFIN/mechanism vs. the Mo-PAE curve
    linearly interpolated (fixed weight t, from point-estimate utility)
    between (lo_point, hi_point), over `users` (already the exact target
    subgroup, e.g. decile-0 users only, or the full common population).
    resample_seeds=False reproduces the "seeds averaged, not resampled"
    validation mode."""
    mech_util_point = mech_point["util"].mean()
    lo_util_point = lo_point["util"].mean()
    hi_util_point = hi_point["util"].mean()
    t = (mech_util_point - lo_util_point) / (hi_util_point - lo_util_point)

    mech_risk = align_risk(mech_point, users)  # (n_seeds_mech, n_users)
    lo_risk = align_risk(lo_point, users)
    hi_risk = align_risk(hi_point, users)
    n_users = len(users)

    # point estimate (seeds averaged, users NOT resampled) -- for the validation check
    mech_pt = mech_risk.mean(axis=0)
    lo_pt = lo_risk.mean(axis=0)
    hi_pt = hi_risk.mean(axis=0)
    curve_pt = (1 - t) * lo_pt + t * hi_pt
    point_delta_mean = float((mech_pt - curve_pt).mean())

    deltas = []
    for _ in range(n_boot):
        user_boot = rng.integers(0, n_users, n_users)  # (a) ONE draw, used for both arms
        if resample_seeds:
            seed_boot_mech = rng.integers(0, mech_risk.shape[0], mech_risk.shape[0])
            # (c) same seed draw for both lambda3 endpoints when they share a seed count
            if lo_risk.shape[0] == hi_risk.shape[0]:
                seed_boot_lo = seed_boot_hi = rng.integers(0, lo_risk.shape[0], lo_risk.shape[0])
            else:
                seed_boot_lo = rng.integers(0, lo_risk.shape[0], lo_risk.shape[0])
                seed_boot_hi = rng.integers(0, hi_risk.shape[0], hi_risk.shape[0])
            mech_b = mech_risk[seed_boot_mech][:, user_boot].mean(axis=0)
            lo_b = lo_risk[seed_boot_lo][:, user_boot].mean(axis=0)
            hi_b = hi_risk[seed_boot_hi][:, user_boot].mean(axis=0)
        else:
            mech_b = mech_risk.mean(axis=0)[user_boot]
            lo_b = lo_risk.mean(axis=0)[user_boot]
            hi_b = hi_risk.mean(axis=0)[user_boot]
        curve_b = (1 - t) * lo_b + t * hi_b
        diff_per_user = mech_b - curve_b  # (d) per-user difference
        deltas.append(diff_per_user.mean())  # then aggregate

    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    # seed-level comparison: per-seed metric MEANS (over `users`), min/max across seeds --
    # NOT utility, the metric itself (e.g. decile0 risk) at full un-resampled users.
    mech_per_seed_means = mech_risk.mean(axis=1)  # (n_seeds_mech,)
    lo_per_seed_means = lo_risk.mean(axis=1)
    hi_per_seed_means = hi_risk.mean(axis=1)
    curve_per_seed_means = np.concatenate([lo_per_seed_means, hi_per_seed_means])  # pooled bracket range

    return dict(
        mean_delta=float(deltas.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        t=float(t), lo_lambda3=lo_point["source"], hi_lambda3=hi_point["source"],
        point_delta_seeds_averaged=point_delta_mean,
        mech_seed_min=float(mech_per_seed_means.min()), mech_seed_max=float(mech_per_seed_means.max()),
        curve_seed_min=float(curve_per_seed_means.min()), curve_seed_max=float(curve_per_seed_means.max()),
        n_users=n_users,
    )


def paired_bootstrap_pairs_matched(mech_point, lo_point, hi_point, users, i_top, j_top, rng,
                                    n_boot=N_BOOT):
    """Same corrected-pairing design as paired_bootstrap_matched, but for
    D_abs/D_rel: the bootstrap unit is PAIRS (the fixed top-10%-similar
    set, indices into `users`), not individual users -- same rationale
    as Task 1's three_fairness_measures. Still: ONE pair-resample draw
    per replicate, applied to all three risk arrays (mech, lo, hi); ONE
    shared seed draw for the two lambda3 endpoints."""
    mech_util_point = mech_point["util"].mean()
    lo_util_point = lo_point["util"].mean()
    hi_util_point = hi_point["util"].mean()
    t = (mech_util_point - lo_util_point) / (hi_util_point - lo_util_point)

    mech_risk = align_risk(mech_point, users)
    lo_risk = align_risk(lo_point, users)
    hi_risk = align_risk(hi_point, users)
    n_top = len(i_top)

    abs_deltas, rel_deltas = [], []
    for _ in range(n_boot):
        pair_boot = rng.integers(0, n_top, n_top)  # ONE draw, shared across all three arms
        seed_boot_mech = rng.integers(0, mech_risk.shape[0], mech_risk.shape[0])
        if lo_risk.shape[0] == hi_risk.shape[0]:
            seed_boot_lo = seed_boot_hi = rng.integers(0, lo_risk.shape[0], lo_risk.shape[0])
        else:
            seed_boot_lo = rng.integers(0, lo_risk.shape[0], lo_risk.shape[0])
            seed_boot_hi = rng.integers(0, hi_risk.shape[0], hi_risk.shape[0])

        def arm_stats(risk, seed_boot):
            r = risk[seed_boot].mean(axis=0)  # (n_users,)
            a = np.abs(r[i_top[pair_boot]] - r[j_top[pair_boot]]).mean()
            mean_r = r.mean()
            return a, (a / mean_r if mean_r > 0 else np.nan)

        mech_a, mech_rel = arm_stats(mech_risk, seed_boot_mech)
        lo_a, lo_rel = arm_stats(lo_risk, seed_boot_lo)
        hi_a, hi_rel = arm_stats(hi_risk, seed_boot_hi)
        curve_a = (1 - t) * lo_a + t * hi_a
        curve_rel = (1 - t) * lo_rel + t * hi_rel
        abs_deltas.append(mech_a - curve_a)
        rel_deltas.append(mech_rel - curve_rel)

    def summarize(vals):
        vals = np.array([v for v in vals if v == v])
        if len(vals) == 0:
            return dict(mean_delta=None, ci_lo=None, ci_hi=None)
        lo_ci, hi_ci = np.percentile(vals, [2.5, 97.5])
        return dict(mean_delta=float(vals.mean()), ci_lo=float(lo_ci), ci_hi=float(hi_ci))

    return dict(D_abs=summarize(abs_deltas), D_rel=summarize(rel_deltas), t=float(t))
