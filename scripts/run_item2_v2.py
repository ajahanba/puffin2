#!/usr/bin/env python
"""Item 2 REDO driver (PI correction, 2026-11-09): rebuilds all 47
matched-utility comparisons with the corrected paired bootstrap
(scripts/paired_bootstrap_v2.py) -- one user-index draw per replicate
applied to both arms, decile membership fixed from the CSV, encoder
seeds resampled with the same draw for both lambda3 endpoints, lambda3
bracket found by walking the sweep in its own order (not sorting by
realized utility, which breaks on GeoLife's near-flat low-lambda3
region). Adds a seed-level comparison column (min/max of per-seed means
for each arm).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm
from scripts.paired_bootstrap_v2 import (
    load_point, find_bracket, paired_bootstrap_matched, paired_bootstrap_pairs_matched,
    MIN_WINDOWS, N_BOOT, TOP_FRAC, SIM_BASES,
)

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
TASK2_BASE = pathlib.Path("reports/figures/task2_formulations")
GI_DP_BASE = pathlib.Path("reports/figures/gi_dp")
PAPER_BASE = pathlib.Path("reports/figures/paper")
SIM_CACHE_BASE = pathlib.Path("reports/figures/individual_fairness")

LAMBDA3_GEOLIFE = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3]
LAMBDA3_STD = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8]

RNG = np.random.default_rng(0)
rows = []


def load_sims(ds):
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
            key = list(loaded.keys())[0]
            sims[b] = loaded[key]
    return user_ids, sims


def geolife_curve():
    names, lps = [], []
    for lp in LAMBDA3_GEOLIFE:
        if lp == 0.3:
            name = "geolife_plain_lambda0.0_10seed"
        elif lp == 2:
            name = "geolife_plain_lp2_10seed_corrected"
        elif lp == 3:
            corrected = RETRAIN_BASE / "geolife_plain_lp3_10seed_corrected"
            name = "geolife_plain_lp3_10seed_corrected" if corrected.exists() else "geolife_plain_lp3_10seed"
        elif lp == 1.2:
            name = "geolife_plain_lp1.2_10seed"
        else:
            name = f"geolife_plain_lp{lp}"
        d = RETRAIN_BASE / name
        if d.exists():
            names.append(d)
            lps.append(lp)
    return [load_point(d) for d in names], lps


def std_curve(ds):
    names, lps = [], []
    for lp in LAMBDA3_STD:
        d = RETRAIN_BASE / f"{ds}_plain_lp{lp}"
        if d.exists():
            names.append(d)
            lps.append(lp)
    return [load_point(d) for d in names], lps


def process_comparison(ds, label, mech_dir, curve_points, lambda3_values, sims, user_ids):
    if not (mech_dir / "summary.json").exists():
        print(f"SKIP missing: {ds}/{label}", flush=True)
        return
    mech = load_point(mech_dir)
    mech_util = mech["util"].mean()
    bracket = find_bracket(mech_util, curve_points, lambda3_values)
    if bracket is None:
        print(f"SKIP (extrapolation required): {ds}/{label} util={mech_util:.4f}", flush=True)
        rows.append(dict(dataset=ds, comparison=label, metric="ALL", unresolvable_extrapolation=True))
        return
    lo, hi, lo_u, hi_u = bracket

    common_users = sorted(set(mech["df"].index) & set(lo["df"].index) & set(hi["df"].index))
    if len(common_users) < 20:
        print(f"SKIP (too few common users={len(common_users)}): {ds}/{label}", flush=True)
        return

    # --- population-level: mean_risk ---
    res = paired_bootstrap_matched(mech, lo, hi, common_users, RNG, n_boot=N_BOOT)
    rows.append(dict(dataset=ds, comparison=label, metric="mean_risk", **res))

    # --- decile-specific: decile0, decile9 (bootstrap WITHIN the subgroup) ---
    for dec, dec_name in ((0, "decile0"), (9, "decile9")):
        dec_users = mech["df"][mech["df"].entropy_decile == dec].index.tolist()
        dec_common = sorted(set(dec_users) & set(common_users))
        if len(dec_common) < 5:
            print(f"  SKIP {dec_name} (too few users={len(dec_common)}): {ds}/{label}", flush=True)
            continue
        res_d = paired_bootstrap_matched(mech, lo, hi, dec_common, RNG, n_boot=N_BOOT)
        rows.append(dict(dataset=ds, comparison=label, metric=dec_name, **res_d))

    # --- D_abs/D_rel, all 4 similarity bases, over the full common-user population ---
    idx_map = {u: i for i, u in enumerate(user_ids)}
    keep_idx = np.array([idx_map[u] for u in common_users if u in idx_map])
    for basis in SIM_BASES:
        sim_sub = sims[basis][np.ix_(keep_idx, keep_idx)]
        i_top, j_top = ifm.select_similar_pairs(sim_sub, top_frac=TOP_FRAC)
        res_pairs = paired_bootstrap_pairs_matched(mech, lo, hi, common_users, i_top, j_top, RNG, n_boot=N_BOOT)
        rows.append(dict(dataset=ds, comparison=label, metric=f"D_abs_{basis}", n_users=len(common_users),
                          t=res_pairs["t"], lo_lambda3=lo["source"], hi_lambda3=hi["source"],
                          **res_pairs["D_abs"]))
        rows.append(dict(dataset=ds, comparison=label, metric=f"D_rel_{basis}", n_users=len(common_users),
                          t=res_pairs["t"], lo_lambda3=lo["source"], hi_lambda3=hi["source"],
                          **res_pairs["D_rel"]))

    print(f"[{ds}] {label}: bracket=({lo['source']},{hi['source']}) t={res['t']:.3f} "
          f"mean_risk_delta={res['mean_delta']:+.4f}[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]", flush=True)


# ---------------- GeoLife ----------------
geo_curve, geo_lps = geolife_curve()
user_ids, sims = load_sims("geolife")
geo_mechs = []
for l4 in (0.3, 1.0, 3.0):
    geo_mechs.append((f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"geolife_ed_soft_lambda{l4}_10seed"))
for l4 in (0.3, 1.0, 3.0):
    seed_suffix = "10seed" if l4 in (0.3, 1.0) else "3seed"
    geo_mechs.append((f"ed_soft_location_gradmatched_lambda{l4}",
                      PAPER_BASE / f"geolife_ed_soft_location_gradmatched_lambda{l4}_{seed_suffix}"))
for basis in ("entropy", "location_distribution"):
    for l4 in (0.3, 1.0, 3.0):
        geo_mechs.append((f"lipschitz_{basis}_lambda{l4}", TASK2_BASE / f"geolife_lipschitz_{basis}_lambda{l4}_10seed"))
for l4 in (0.3, 1.0):
    geo_mechs.append((f"group_dro_lambda{l4}", TASK2_BASE / f"geolife_group_dro_lambda{l4}_3seed"))
for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
    geo_mechs.append((f"gi_dp_eps{eps}", GI_DP_BASE / f"geolife_gi_dp_eps{eps}"))
for label, d in geo_mechs:
    process_comparison("geolife", label, d, geo_curve, geo_lps, sims, user_ids)

# base table
base_idx = geo_lps.index(0.3)
process_comparison("geolife", "base_table_no_privacy_vs_plain0.3", RETRAIN_BASE / "geolife_no_privacy",
                    geo_curve, geo_lps, sims, user_ids)

# ---------------- Foursquare ----------------
fsq_curve, fsq_lps = std_curve("foursquare")
user_ids, sims = load_sims("foursquare")
fsq_mechs = []
for l4 in (0.3, 1.0, 3.0):
    fsq_mechs.append((f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"foursquare_ed_soft_lambda{l4}"))
for basis in ("entropy", "location_distribution"):
    for l4 in (0.3, 1.0, 3.0):
        fsq_mechs.append((f"lipschitz_{basis}_lambda{l4}", TASK2_BASE / f"foursquare_lipschitz_{basis}_lambda{l4}_3seed"))
for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
    fsq_mechs.append((f"gi_dp_eps{eps}", GI_DP_BASE / f"foursquare_gi_dp_eps{eps}"))
for label, d in fsq_mechs:
    process_comparison("foursquare", label, d, fsq_curve, fsq_lps, sims, user_ids)

# ---------------- Gowalla-SF-Bay ----------------
gow_curve, gow_lps = std_curve("gowalla_sf_bay")
user_ids, sims = load_sims("gowalla_sf_bay")
gow_mechs = []
for l4 in (0.3, 1.0, 3.0):
    gow_mechs.append((f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"gowalla_sf_bay_ed_soft_lambda{l4}"))
for basis in ("entropy", "location_distribution"):
    for l4 in (0.3, 1.0, 3.0):
        gow_mechs.append((f"lipschitz_{basis}_lambda{l4}", TASK2_BASE / f"gowalla_sf_bay_lipschitz_{basis}_lambda{l4}_3seed"))
for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
    gow_mechs.append((f"gi_dp_eps{eps}", GI_DP_BASE / f"gowalla_sf_bay_gi_dp_eps{eps}"))
for label, d in gow_mechs:
    process_comparison("gowalla_sf_bay", label, d, gow_curve, gow_lps, sims, user_ids)

out_df = pd.DataFrame(rows)
out_path = "reports/figures/paper/item2_bootstrap_cis_matched_utility.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}, {len(out_df)} rows")
