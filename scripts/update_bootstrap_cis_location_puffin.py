#!/usr/bin/env python
"""Recomputes the ed_soft_location_gradmatched_lambda{0.3,1.0} matched-
utility comparison rows in item2_bootstrap_cis_matched_utility.csv
against the 10-seed runs (reports/figures/paper/geolife_ed_soft_
location_gradmatched_lambda{0.3,1.0}_10seed), using the exact same
bootstrap procedure as every other row (scripts/paired_bootstrap_v2.py's
paired_bootstrap_matched/paired_bootstrap_pairs_matched against the
interpolated Mo-PAE curve). The previously-committed rows for these two
comparisons were computed against the stale _3seed directories (scripts/
run_item2_v2.py's mechanism list pointed there), confirmed by
reproducing their exact values using the _3seed path; this recomputes
against the correct 10-seed data and replaces only those 22 rows.
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
    N_BOOT, TOP_FRAC, SIM_BASES,
)

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
PAPER_BASE = pathlib.Path("reports/figures/paper")
SIM_CACHE_BASE = pathlib.Path("reports/figures/individual_fairness")
OUT_PATH = "reports/figures/paper/item2_bootstrap_cis_matched_utility.csv"

LAMBDA3_GEOLIFE = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3]
RNG = np.random.default_rng(0)


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


def process_comparison(ds, label, mech_dir, curve_points, lambda3_values, sims, user_ids, rows):
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

    res = paired_bootstrap_matched(mech, lo, hi, common_users, RNG, n_boot=N_BOOT)
    rows.append(dict(dataset=ds, comparison=label, metric="mean_risk", **res))

    for dec, dec_name in ((0, "decile0"), (9, "decile9")):
        dec_users = mech["df"][mech["df"].entropy_decile == dec].index.tolist()
        dec_common = sorted(set(dec_users) & set(common_users))
        if len(dec_common) < 5:
            print(f"  SKIP {dec_name} (too few users={len(dec_common)}): {ds}/{label}", flush=True)
            continue
        res_d = paired_bootstrap_matched(mech, lo, hi, dec_common, RNG, n_boot=N_BOOT)
        rows.append(dict(dataset=ds, comparison=label, metric=dec_name, **res_d))

    idx_map = {u: i for i, u in enumerate(user_ids)}
    keep_idx = np.array([idx_map[u] for u in common_users if u in idx_map])
    assert len(keep_idx) == len(common_users), (
        f"{ds}/{label}: {len(common_users) - len(keep_idx)} common_users missing from the sims cache")
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

    print(f"[{ds}] {label}: bracket=({lo['source']},{hi['source']}) mech_dir={mech_dir} "
          f"mean_risk_delta={res['mean_delta']:+.4f}[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]", flush=True)


geo_curve, geo_lps = geolife_curve()
user_ids, sims = load_sims("geolife")

new_rows = []
for l4 in (0.3, 1.0):
    label = f"ed_soft_location_gradmatched_lambda{l4}"
    mech_dir = PAPER_BASE / f"geolife_ed_soft_location_gradmatched_lambda{l4}_10seed"
    process_comparison("geolife", label, mech_dir, geo_curve, geo_lps, sims, user_ids, new_rows)

new_df = pd.DataFrame(new_rows)
old_df = pd.read_csv(OUT_PATH)
replaced_labels = {f"ed_soft_location_gradmatched_lambda{l4}" for l4 in (0.3, 1.0)}
n_before = len(old_df)
kept_df = old_df[~((old_df["dataset"] == "geolife") & (old_df["comparison"].isin(replaced_labels)))]
n_removed = n_before - len(kept_df)
out_df = pd.concat([kept_df, new_df], ignore_index=True)
out_df.to_csv(OUT_PATH, index=False)
print(f"\nRemoved {n_removed} old (stale _3seed-sourced) rows for {sorted(replaced_labels)}, "
      f"added {len(new_df)} new rows (sourced from _10seed). Wrote {OUT_PATH}, {len(out_df)} total rows.")
