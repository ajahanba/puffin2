#!/usr/bin/env python
"""Computes the base-table row as a DIRECT point comparison (no-privacy
vs. plain lambda3=0.3 exactly, no interpolation), and replaces those
rows in the CSV. The base table is a direct comparison, unlike the
matched-utility table's curve-interpolated comparisons."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm
from scripts.bootstrap_cis import load_point, align_risk, MIN_WINDOWS, N_BOOT, TOP_FRAC, SIM_BASES

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
SIM_CACHE_BASE = pathlib.Path("reports/figures/individual_fairness")
RNG = np.random.default_rng(1)  # different stream than the main run, independent of it


def direct_bootstrap(a_point, b_point, users, rng, n_boot=N_BOOT):
    """delta = a - b, direct two-point paired bootstrap (no interpolation)."""
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
    a_seed_means = a_risk.mean(axis=1)
    b_seed_means = b_risk.mean(axis=1)
    return dict(mean_delta=float(deltas.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                mech_seed_min=float(a_seed_means.min()), mech_seed_max=float(a_seed_means.max()),
                curve_seed_min=float(b_seed_means.min()), curve_seed_max=float(b_seed_means.max()))


def direct_pairs_bootstrap(a_point, b_point, users, i_top, j_top, rng, n_boot=N_BOOT):
    a_risk = align_risk(a_point, users)
    b_risk = align_risk(b_point, users)
    n_top = len(i_top)
    abs_deltas, rel_deltas = [], []
    for _ in range(n_boot):
        pair_boot = rng.integers(0, n_top, n_top)
        seed_boot_a = rng.integers(0, a_risk.shape[0], a_risk.shape[0])
        seed_boot_b = rng.integers(0, b_risk.shape[0], b_risk.shape[0])

        def stats(risk, seed_boot):
            r = risk[seed_boot].mean(axis=0)
            a = np.abs(r[i_top[pair_boot]] - r[j_top[pair_boot]]).mean()
            mean_r = r.mean()
            return a, (a / mean_r if mean_r > 0 else np.nan)

        a_a, a_rel = stats(a_risk, seed_boot_a)
        b_a, b_rel = stats(b_risk, seed_boot_b)
        abs_deltas.append(a_a - b_a)
        rel_deltas.append(a_rel - b_rel)

    def summarize(vals):
        vals = np.array([v for v in vals if v == v])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return dict(mean_delta=float(vals.mean()), ci_lo=float(lo), ci_hi=float(hi))

    return dict(D_abs=summarize(abs_deltas), D_rel=summarize(rel_deltas))


no_priv = load_point(RETRAIN_BASE / "geolife_no_privacy")
plain03 = load_point(RETRAIN_BASE / "geolife_plain_lambda0.0_10seed")
common_users = sorted(set(no_priv["df"].index) & set(plain03["df"].index))
print("n common users:", len(common_users))

new_rows = []
res = direct_bootstrap(plain03, no_priv, common_users, RNG, n_boot=N_BOOT)  # delta = plain0.3 - no_privacy
new_rows.append(dict(dataset="geolife", comparison="base_table_no_privacy_vs_plain0.3", metric="mean_risk", **res))

for dec, dec_name in ((0, "decile0"), (9, "decile9")):
    dec_users = sorted(set(plain03["df"][plain03["df"].entropy_decile == dec].index) & set(common_users))
    if len(dec_users) < 5:
        continue
    res_d = direct_bootstrap(plain03, no_priv, dec_users, RNG, n_boot=N_BOOT)
    new_rows.append(dict(dataset="geolife", comparison="base_table_no_privacy_vs_plain0.3", metric=dec_name, **res_d))

user_ids = __import__("json").loads((SIM_CACHE_BASE / "geolife_user_ids.json").read_text())
sims = {b: np.load(SIM_CACHE_BASE / "geolife_similarities.npz")[b] for b in SIM_BASES}
idx_map = {u: i for i, u in enumerate(user_ids)}
keep_idx = np.array([idx_map[u] for u in common_users if u in idx_map])
for basis in SIM_BASES:
    sim_sub = sims[basis][np.ix_(keep_idx, keep_idx)]
    i_top, j_top = ifm.select_similar_pairs(sim_sub, top_frac=TOP_FRAC)
    res_p = direct_pairs_bootstrap(plain03, no_priv, common_users, i_top, j_top, RNG, n_boot=N_BOOT)
    new_rows.append(dict(dataset="geolife", comparison="base_table_no_privacy_vs_plain0.3",
                          metric=f"D_abs_{basis}", n_users=len(common_users), **res_p["D_abs"]))
    new_rows.append(dict(dataset="geolife", comparison="base_table_no_privacy_vs_plain0.3",
                          metric=f"D_rel_{basis}", n_users=len(common_users), **res_p["D_rel"]))

for r in new_rows:
    print(r)

path = "reports/figures/paper/bootstrap_cis_matched_utility.csv"
df = pd.read_csv(path)
df = df[df.comparison != "base_table_no_privacy_vs_plain0.3"]
new_df = pd.DataFrame(new_rows)
out = pd.concat([df, new_df], ignore_index=True)
out.to_csv(path, index=False)
print(f"\nFixed base table, rewrote {path}, {len(out)} rows total")
