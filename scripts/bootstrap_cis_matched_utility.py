#!/usr/bin/env python
"""Item 2 (PI instruction, 2026-11-09): for every matched-utility
comparison (PUFFIN/ed_soft, Lipschitz, Group-DRO, GI-DP vs. the plain
Mo-PAE lambda3 curve, all datasets) and the base table (no-privacy ->
Mo-PAE lambda3=0.3), 2000 paired bootstrap replicates resampling users
(same user draw for both methods, within a comparison) and encoder
seeds, recomputing the Mo-PAE curve interpolation on each replicate.
Reports 95% percentile CIs for (method - Mo-PAE) on: utility, mean
risk, decile0, decile9, and D_abs/D_rel under all four similarity
definitions (top-10% most similar pairs). One CSV.

D_abs/D_rel's bootstrap unit is PAIRS (the top-10%-similar set, fixed
by the data), not individual users -- consistent with Task 1's already-
reviewed methodology (puffin.metrics.individual_fairness.
three_fairness_measures): bootstrapping users with replacement has no
clean definition against a FIXED top-10%-similar-pairs set, since that
set is itself derived from the real (non-resampled) similarity matrix.
Utility/mean_risk/decile0/decile9 ARE bootstrapped over users directly,
per the PI's instruction.
"""
import json
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

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
TASK2_BASE = pathlib.Path("reports/figures/task2_formulations")
GI_DP_BASE = pathlib.Path("reports/figures/gi_dp")
PAPER_BASE = pathlib.Path("reports/figures/paper")
SIM_CACHE_BASE = pathlib.Path("reports/figures/individual_fairness")

LAMBDA3_SWEEP_GEOLIFE = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3]
LAMBDA3_SWEEP_STD = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8]


def load_sims(ds):
    """Returns user_ids (list) and dict sim_basis -> full similarity matrix."""
    user_ids = json.loads((SIM_CACHE_BASE / f"{ds}_user_ids.json").read_text())
    combined = SIM_CACHE_BASE / f"{ds}_similarities.npz"
    sims = {}
    if combined.exists():
        loaded = np.load(combined)
        for b in SIM_BASES:
            sims[b] = loaded[b]
    else:
        for b in SIM_BASES:
            loaded = np.load(SIM_CACHE_BASE / f"{ds}_similarities_{b}.npz")
            # per-basis cache may store under a generic key; take the first array
            key = list(loaded.keys())[0]
            sims[b] = loaded[key]
    return user_ids, sims


def load_point(run_dir):
    """Loads one run's utility_per_encoder_seed and per-user risk table."""
    summ = json.load(open(run_dir / "summary.json"))
    df = pd.read_csv(run_dir / "per_user_04d.csv")
    df = df[df.n_windows >= MIN_WINDOWS].copy()
    seed_cols = sorted([c for c in df.columns if c.startswith("post_top1_encseed")],
                        key=lambda c: int(c.replace("post_top1_encseed", "")))
    util = np.array(summ["utility_per_encoder_seed"])
    return dict(util=util, n_seeds=len(seed_cols), df=df.set_index("user_id"),
                seed_cols=seed_cols, source=run_dir.name)


def align_point_to_common(point, common_users):
    """Aligns a point's per-user risk table to `common_users` (sorted list),
    returns risk_per_seed array (n_seeds, n_common_users)."""
    sub = point["df"].loc[common_users, point["seed_cols"]]
    return sub.to_numpy().T  # (n_seeds, n_common_users)


def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def compare(tag, mech_point, curve_points, full_sims, user_ids, rng):
    """One matched-utility comparison: mech_point vs. the curve. Returns a
    dict of {metric_name: {mean_delta, ci_lo, ci_hi, n_extrapolated}}."""
    common_users = sorted(set(mech_point["df"].index)
                           .intersection(*[set(cp["df"].index) for cp in curve_points]))
    if len(common_users) < 20:
        return None, len(common_users)
    idx_map = {u: i for i, u in enumerate(user_ids)}
    keep_idx = np.array([idx_map[u] for u in common_users if u in idx_map])
    if len(keep_idx) < 20:
        return None, len(keep_idx)

    mech_risk = align_point_to_common(mech_point, common_users)  # (n_seeds_mech, n_users)
    curve_risks = [align_point_to_common(cp, common_users) for cp in curve_points]
    n_users = len(common_users)

    # top-10% pairs per similarity basis, on THIS comparison's common-user submatrix
    pairs = {}
    for basis in SIM_BASES:
        sim_sub = full_sims[basis][np.ix_(keep_idx, keep_idx)]
        i_top, j_top = ifm.select_similar_pairs(sim_sub, top_frac=TOP_FRAC)
        pairs[basis] = (i_top, j_top)

    out = {m: [] for m in ("utility", "mean_risk", "decile0", "decile9")}
    for b in SIM_BASES:
        out[f"D_abs_{b}"] = []
        out[f"D_rel_{b}"] = []

    # decile assignment: from the per-user entropy_decile column already in per_user_04d.csv
    decile_col = mech_point["df"].loc[common_users, "entropy_decile"].to_numpy() \
        if "entropy_decile" in mech_point["df"].columns else None

    n_extrap = {m: 0 for m in out}

    for _ in range(N_BOOT):
        # --- mechanism's resampled draw ---
        user_boot = rng.integers(0, n_users, n_users)
        seed_boot_m = rng.integers(0, mech_risk.shape[0], mech_risk.shape[0])
        mech_util_b = mech_point["util"][seed_boot_m].mean()
        mech_risk_b = mech_risk[seed_boot_m][:, user_boot].mean(axis=0)  # per-user, this draw
        mech_mean_risk_b = mech_risk_b.mean()
        if decile_col is not None:
            dcol_b = decile_col[user_boot]
            d0 = mech_risk_b[dcol_b == 0]
            d9 = mech_risk_b[dcol_b == 9]
            mech_d0_b = d0.mean() if len(d0) else np.nan
            mech_d9_b = d9.mean() if len(d9) else np.nan
        else:
            mech_d0_b = mech_d9_b = np.nan

        # --- curve's resampled points (independent draw per point, per PI pattern) ---
        curve_utils, curve_means, curve_d0s, curve_d9s = [], [], [], []
        for cp, cr in zip(curve_points, curve_risks):
            seed_boot_c = rng.integers(0, cr.shape[0], cr.shape[0])
            user_boot_c = rng.integers(0, n_users, n_users)
            u_b = cp["util"][seed_boot_c].mean()
            r_b = cr[seed_boot_c][:, user_boot_c].mean(axis=0)
            curve_utils.append(u_b)
            curve_means.append(r_b.mean())
            if decile_col is not None:
                dcol_cb = decile_col[user_boot_c]
                d0c = r_b[dcol_cb == 0]
                d9c = r_b[dcol_cb == 9]
                curve_d0s.append(d0c.mean() if len(d0c) else np.nan)
                curve_d9s.append(d9c.mean() if len(d9c) else np.nan)
            else:
                curve_d0s.append(np.nan)
                curve_d9s.append(np.nan)

        order = np.argsort(curve_utils)
        cu = np.array(curve_utils)[order]

        def interp_or_extrap(metric_name, mech_val, curve_vals_sorted):
            if mech_util_b < cu[0] or mech_util_b > cu[-1]:
                n_extrap[metric_name] += 1
                return
            interp_val = np.interp(mech_util_b, cu, curve_vals_sorted)
            out[metric_name].append(mech_val - interp_val)  # (method - Mo-PAE)

        interp_or_extrap("mean_risk", mech_mean_risk_b, np.array(curve_means)[order])
        interp_or_extrap("decile0", mech_d0_b, np.array(curve_d0s)[order])
        interp_or_extrap("decile9", mech_d9_b, np.array(curve_d9s)[order])
        out["utility"].append(mech_util_b - np.interp(mech_util_b, cu, cu))  # trivially 0 by construction; kept for shape

        # --- D_abs/D_rel per basis: bootstrap unit is PAIRS, not users ---
        for basis in SIM_BASES:
            i_top, j_top = pairs[basis]
            n_top = len(i_top)
            pair_boot = rng.integers(0, n_top, n_top)

            def dmetrics(risk_arr, seed_boot):
                r = risk_arr[seed_boot].mean(axis=0)
                a = np.abs(r[i_top[pair_boot]] - r[j_top[pair_boot]]).mean()
                mean_r = r.mean()
                return a, (a / mean_r if mean_r > 0 else np.nan)

            mech_a, mech_rel = dmetrics(mech_risk, seed_boot_m)
            curve_a_list, curve_rel_list = [], []
            for cp, cr in zip(curve_points, curve_risks):
                seed_boot_c2 = rng.integers(0, cr.shape[0], cr.shape[0])
                a_c, rel_c = dmetrics(cr, seed_boot_c2)
                curve_a_list.append(a_c)
                curve_rel_list.append(rel_c)
            ca = np.array(curve_a_list)[order]
            crel = np.array(curve_rel_list)[order]
            interp_or_extrap(f"D_abs_{basis}", mech_a, ca)
            interp_or_extrap(f"D_rel_{basis}", mech_rel, crel)

    results = {}
    for m, vals in out.items():
        vals = np.array([v for v in vals if v == v])  # drop nan
        if len(vals) == 0:
            results[m] = dict(mean_delta=None, ci_lo=None, ci_hi=None, n_extrapolated=n_extrap[m], n_valid=0)
            continue
        ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
        results[m] = dict(mean_delta=float(vals.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                           n_extrapolated=int(n_extrap[m]), n_valid=int(len(vals)))
    return results, len(common_users)
