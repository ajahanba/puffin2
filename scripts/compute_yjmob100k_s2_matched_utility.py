#!/usr/bin/env python
"""Matched-utility comparison for the YJMob100K SECOND independent
2000-user sample (seed=1): PUFFIN ed_soft lambda4=1.0 vs. the plain
Mo-PAE lambda3 in {0.1, 0.2} bracket (same pair that bracketed it in
the first sample). Metrics: mean risk, decile 0/9, and D_abs under
entropy / trajectory_shape / radius_of_gyration (not D_rel, not
location_distribution -- narrower than the standard 4-basis/D_abs+
D_rel sweep used elsewhere, per this task's explicit scope). Same
procedure as scripts/paired_bootstrap_v2.py / run_item2_v2_gap_fill.py,
with the curve built directly from just these 2 points (the full
lambda3 sweep was not run for this sample by design). Appended to the
CSV under dataset="yjmob100k_s2" so it cannot collide with the first
sample's existing "yjmob100k" rows."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm
from scripts.paired_bootstrap_v2 import (
    load_point, find_bracket, paired_bootstrap_matched, paired_bootstrap_pairs_matched,
    N_BOOT, TOP_FRAC,
)

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
SAMPLE_BASE = pathlib.Path("reports/figures/yjmob100k_sample_s2")
OUT_PATH = "reports/figures/paper/item2_bootstrap_cis_matched_utility.csv"
BASES = ("entropy", "trajectory_shape", "radius_of_gyration")
RNG = np.random.default_rng(3)
DATASET = "yjmob100k_s2"
LABEL = "ed_soft_entropy_lambda1.0"

user_ids = json.loads((SAMPLE_BASE / "user_ids.json").read_text())
loaded = np.load(SAMPLE_BASE / "similarities.npz")
sims = {b: loaded[b] for b in BASES}

lp01 = load_point(RETRAIN_BASE / "yjmob100k_s2_plain_lp0.1")
lp02 = load_point(RETRAIN_BASE / "yjmob100k_s2_plain_lp0.2")
mech = load_point(RETRAIN_BASE / "yjmob100k_s2_ed_soft_lambda1.0")
mech_util = mech["util"].mean()

bracket = find_bracket(mech_util, [lp01, lp02], [0.1, 0.2])
assert bracket is not None, f"lambda3 in {{0.1,0.2}} do not bracket ed_soft's utility ({mech_util:.4f}) for this sample"
lo, hi, lo_u, hi_u = bracket
print(f"bracket: lo={lo['source']} (util={lo_u:.4f}), hi={hi['source']} (util={hi_u:.4f}), "
      f"mech util={mech_util:.4f}", flush=True)

common_users = sorted(set(mech["df"].index) & set(lo["df"].index) & set(hi["df"].index))
print(f"n_common_users={len(common_users)}", flush=True)

rows = []
res = paired_bootstrap_matched(mech, lo, hi, common_users, RNG, n_boot=N_BOOT)
rows.append(dict(dataset=DATASET, comparison=LABEL, metric="mean_risk", **res))
print(f"mean_risk: delta={res['mean_delta']:+.4f} [{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]", flush=True)

for dec, dec_name in ((0, "decile0"), (9, "decile9")):
    dec_users = mech["df"][mech["df"].entropy_decile == dec].index.tolist()
    dec_common = sorted(set(dec_users) & set(common_users))
    if len(dec_common) < 5:
        print(f"SKIP {dec_name} (too few users={len(dec_common)})", flush=True)
        continue
    res_d = paired_bootstrap_matched(mech, lo, hi, dec_common, RNG, n_boot=N_BOOT)
    rows.append(dict(dataset=DATASET, comparison=LABEL, metric=dec_name, **res_d))
    print(f"{dec_name}: delta={res_d['mean_delta']:+.4f} [{res_d['ci_lo']:+.4f},{res_d['ci_hi']:+.4f}]", flush=True)

idx_map = {u: i for i, u in enumerate(user_ids)}
keep_idx = np.array([idx_map[u] for u in common_users if u in idx_map])
assert len(keep_idx) == len(common_users), (
    f"{len(common_users) - len(keep_idx)} common_users missing from the sims cache")
for basis in BASES:
    sim_sub = sims[basis][np.ix_(keep_idx, keep_idx)]
    i_top, j_top = ifm.select_similar_pairs(sim_sub, top_frac=TOP_FRAC)
    res_pairs = paired_bootstrap_pairs_matched(mech, lo, hi, common_users, i_top, j_top, RNG, n_boot=N_BOOT)
    rows.append(dict(dataset=DATASET, comparison=LABEL, metric=f"D_abs_{basis}", n_users=len(common_users),
                      t=res_pairs["t"], lo_lambda3=lo["source"], hi_lambda3=hi["source"],
                      **res_pairs["D_abs"]))
    print(f"D_abs_{basis}: delta={res_pairs['D_abs']['mean_delta']:+.4f} "
          f"[{res_pairs['D_abs']['ci_lo']:+.4f},{res_pairs['D_abs']['ci_hi']:+.4f}]", flush=True)

new_df = pd.DataFrame(rows)
old_df = pd.read_csv(OUT_PATH)
already_present = ((old_df["dataset"] == DATASET) & (old_df["comparison"] == LABEL)).any()
if already_present:
    raise AssertionError(f"'{DATASET}/{LABEL}' already exists in the CSV, refusing to duplicate")

out_df = pd.concat([old_df, new_df], ignore_index=True)
out_df.to_csv(OUT_PATH, index=False)
print(f"\nAppended {len(new_df)} new rows for '{DATASET}/{LABEL}'. Wrote {OUT_PATH}, {len(out_df)} total rows "
      f"(was {len(old_df)}).")
