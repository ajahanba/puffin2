#!/usr/bin/env python
"""Item 2 gap-fill (PI instruction, 2026-11-09): (1) add YJMob100K
comparisons (ed_soft lambda4 in {0.3,1.0,3.0} vs. the plain lambda3
curve, same metrics, same corrected paired procedure); (2) add the
base-table comparison (no-privacy vs. plain lambda3=0.3) for
Foursquare, Gowalla-SF-Bay, and YJMob100K (GeoLife's already exists).
Appends to the same CSV, does not touch existing rows.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm
from scripts.paired_bootstrap_v2 import (
    load_point, find_bracket, paired_bootstrap_matched, paired_bootstrap_pairs_matched, align_risk,
    MIN_WINDOWS, N_BOOT, TOP_FRAC, SIM_BASES,
)

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
SIM_CACHE_BASE = pathlib.Path("reports/figures/individual_fairness")
YJMOB_SIM_BASE = pathlib.Path("reports/figures/yjmob100k_sample")

LAMBDA3_STD = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8]
RNG = np.random.default_rng(2)  # fresh stream, independent of the main run/base-table-fix runs
rows = []


def load_sims_generic(user_ids_path, sim_npz_path):
    user_ids = json.loads(user_ids_path.read_text())
    loaded = np.load(sim_npz_path)
    sims = {b: loaded[b] for b in SIM_BASES}
    return user_ids, sims


def std_curve(ds, prefer_corrected=None):
    """prefer_corrected: dict {lp: corrected_dirname} to use instead of the
    raw point when it exists (e.g. YJMob100K's plateau-audit-corrected lp0.8)."""
    prefer_corrected = prefer_corrected or {}
    names, lps = [], []
    for lp in LAMBDA3_STD:
        name = prefer_corrected.get(lp, f"{ds}_plain_lp{lp}")
        d = RETRAIN_BASE / name
        if not d.exists() and lp in prefer_corrected:
            d = RETRAIN_BASE / f"{ds}_plain_lp{lp}"
        if d.exists():
            names.append(d)
            lps.append(lp)
    return [load_point(d) for d in names], lps


def direct_two_point(a_point, b_point, users, rng, n_boot=N_BOOT):
    a_risk = align_risk(a_point, users)
    b_risk = align_risk(b_point, users)
    n_users = len(users)
    deltas = []
    for _ in range(n_boot):
        user_boot = rng.integers(0, n_users, n_users)
        seed_boot_a = rng.integers(0, a_risk.shape[0], a_risk.shape[0])
        seed_boot_b = rng.integers(0, b_risk.shape[0], b_risk.shape[0])
        a_b = a_risk[seed_boot_a][:, user_boot].mean(axis=0)
        b_b = b_risk[seed_boot_b][:, user_boot].mean(axis=0)
        deltas.append((a_b - b_b).mean())
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    a_sm, b_sm = a_risk.mean(axis=1), b_risk.mean(axis=1)
    return dict(mean_delta=float(deltas.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                mech_seed_min=float(a_sm.min()), mech_seed_max=float(a_sm.max()),
                curve_seed_min=float(b_sm.min()), curve_seed_max=float(b_sm.max()))


def direct_two_point_pairs(a_point, b_point, users, i_top, j_top, rng, n_boot=N_BOOT):
    a_risk = align_risk(a_point, users)
    b_risk = align_risk(b_point, users)
    n_top = len(i_top)
    abs_d, rel_d = [], []
    for _ in range(n_boot):
        pair_boot = rng.integers(0, n_top, n_top)
        seed_boot_a = rng.integers(0, a_risk.shape[0], a_risk.shape[0])
        seed_boot_b = rng.integers(0, b_risk.shape[0], b_risk.shape[0])

        def stats(risk, seed_boot):
            r = risk[seed_boot].mean(axis=0)
            a = np.abs(r[i_top[pair_boot]] - r[j_top[pair_boot]]).mean()
            mr = r.mean()
            return a, (a / mr if mr > 0 else np.nan)

        a_a, a_rel = stats(a_risk, seed_boot_a)
        b_a, b_rel = stats(b_risk, seed_boot_b)
        abs_d.append(a_a - b_a)
        rel_d.append(a_rel - b_rel)

    def summarize(vals):
        vals = np.array([v for v in vals if v == v])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return dict(mean_delta=float(vals.mean()), ci_lo=float(lo), ci_hi=float(hi))

    return dict(D_abs=summarize(abs_d), D_rel=summarize(rel_d))


def add_base_table(ds, plain03_dir, no_priv_dir, sims, user_ids):
    plain03 = load_point(plain03_dir)
    no_priv = load_point(no_priv_dir)
    common_users = sorted(set(plain03["df"].index) & set(no_priv["df"].index))
    if len(common_users) < 20:
        print(f"SKIP base table {ds}: too few users ({len(common_users)})", flush=True)
        return
    label = "base_table_no_privacy_vs_plain0.3"
    res = direct_two_point(plain03, no_priv, common_users, RNG, n_boot=N_BOOT)
    rows.append(dict(dataset=ds, comparison=label, metric="mean_risk", **res))
    for dec, dec_name in ((0, "decile0"), (9, "decile9")):
        dec_users = sorted(set(plain03["df"][plain03["df"].entropy_decile == dec].index) & set(common_users))
        if len(dec_users) < 5:
            continue
        res_d = direct_two_point(plain03, no_priv, dec_users, RNG, n_boot=N_BOOT)
        rows.append(dict(dataset=ds, comparison=label, metric=dec_name, **res_d))
    idx_map = {u: i for i, u in enumerate(user_ids)}
    keep_idx = np.array([idx_map[u] for u in common_users if u in idx_map])
    for basis in SIM_BASES:
        sim_sub = sims[basis][np.ix_(keep_idx, keep_idx)]
        i_top, j_top = ifm.select_similar_pairs(sim_sub, top_frac=TOP_FRAC)
        res_p = direct_two_point_pairs(plain03, no_priv, common_users, i_top, j_top, RNG, n_boot=N_BOOT)
        rows.append(dict(dataset=ds, comparison=label, metric=f"D_abs_{basis}", n_users=len(common_users), **res_p["D_abs"]))
        rows.append(dict(dataset=ds, comparison=label, metric=f"D_rel_{basis}", n_users=len(common_users), **res_p["D_rel"]))
    print(f"[{ds}] base table done, n_common_users={len(common_users)}", flush=True)


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
    for basis in SIM_BASES:
        sim_sub = sims[basis][np.ix_(keep_idx, keep_idx)]
        i_top, j_top = ifm.select_similar_pairs(sim_sub, top_frac=TOP_FRAC)
        res_pairs = paired_bootstrap_pairs_matched(mech, lo, hi, common_users, i_top, j_top, RNG, n_boot=N_BOOT)
        rows.append(dict(dataset=ds, comparison=label, metric=f"D_abs_{basis}", n_users=len(common_users),
                          t=res_pairs["t"], lo_lambda3=lo["source"], hi_lambda3=hi["source"], **res_pairs["D_abs"]))
        rows.append(dict(dataset=ds, comparison=label, metric=f"D_rel_{basis}", n_users=len(common_users),
                          t=res_pairs["t"], lo_lambda3=lo["source"], hi_lambda3=hi["source"], **res_pairs["D_rel"]))

    print(f"[{ds}] {label}: bracket=({lo['source']},{hi['source']}) t={res['t']:.3f} "
          f"mean_risk_delta={res['mean_delta']:+.4f}[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]", flush=True)


# ================= (1) YJMob100K comparisons =================
yjmob_user_ids, yjmob_sims = load_sims_generic(YJMOB_SIM_BASE / "user_ids.json", YJMOB_SIM_BASE / "similarities.npz")
yjmob_curve, yjmob_lps = std_curve("yjmob100k", prefer_corrected={0.8: "yjmob100k_plain_lp0.8_corrected"})
print("YJMob100K curve points:", [(cp["source"], lp) for cp, lp in zip(yjmob_curve, yjmob_lps)], flush=True)
for l4 in (0.3, 1.0, 3.0):
    process_comparison("yjmob100k", f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"yjmob100k_ed_soft_lambda{l4}",
                        yjmob_curve, yjmob_lps, yjmob_sims, yjmob_user_ids)

# ================= (2) base tables for Foursquare, Gowalla-SF-Bay, YJMob100K =================
fsq_user_ids, fsq_sims = load_sims_generic(SIM_CACHE_BASE / "foursquare_user_ids.json", SIM_CACHE_BASE / "foursquare_similarities.npz")
add_base_table("foursquare", RETRAIN_BASE / "foursquare_plain_lp0.3", RETRAIN_BASE / "foursquare_no_privacy", fsq_sims, fsq_user_ids)

gow_user_ids = json.loads((SIM_CACHE_BASE / "gowalla_sf_bay_user_ids.json").read_text())
gow_sims = {b: np.load(SIM_CACHE_BASE / f"gowalla_sf_bay_similarities_{b}.npz")[list(np.load(SIM_CACHE_BASE / f"gowalla_sf_bay_similarities_{b}.npz").keys())[0]] for b in SIM_BASES}
add_base_table("gowalla_sf_bay", RETRAIN_BASE / "gowalla_sf_bay_plain_lp0.3", RETRAIN_BASE / "gowalla_sf_bay_no_privacy", gow_sims, gow_user_ids)

add_base_table("yjmob100k", RETRAIN_BASE / "yjmob100k_plain_lp0.3", RETRAIN_BASE / "yjmob100k_no_privacy", yjmob_sims, yjmob_user_ids)

# ================= append to the existing CSV =================
path = "reports/figures/paper/item2_bootstrap_cis_matched_utility.csv"
existing = pd.read_csv(path)
new_df = pd.DataFrame(rows)
out = pd.concat([existing, new_df], ignore_index=True)
out.to_csv(path, index=False)
print(f"\nAppended {len(new_df)} rows, wrote {path}, {len(out)} rows total")
