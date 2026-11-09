#!/usr/bin/env python
"""Driver for item 2 (PI instruction, 2026-11-09). Builds every
matched-utility comparison (PUFFIN, Lipschitz, Group-DRO, GI-DP vs.
plain Mo-PAE) and the base table (no-privacy -> Mo-PAE lambda3=0.3)
across GeoLife/Foursquare/Gowalla-SF-Bay, writes one CSV.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.bootstrap_cis_matched_utility import (
    RETRAIN_BASE, TASK2_BASE, GI_DP_BASE, PAPER_BASE,
    load_sims, load_point, compare, LAMBDA3_SWEEP_GEOLIFE, LAMBDA3_SWEEP_STD,
)

RNG = np.random.default_rng(0)
rows = []


def geolife_curve_points():
    pts = []
    for lp in LAMBDA3_SWEEP_GEOLIFE:
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
            pts.append(load_point(d))
    return pts


def std_curve_points(ds):
    pts = []
    for lp in LAMBDA3_SWEEP_STD:
        name = f"{ds}_plain_lp{lp}"
        d = RETRAIN_BASE / name
        if d.exists():
            pts.append(load_point(d))
    return pts


def run_dataset(ds, curve_points, mech_list):
    """mech_list: list of (label, run_dir) tuples."""
    user_ids, sims = load_sims(ds)
    for label, run_dir in mech_list:
        if not (run_dir / "summary.json").exists():
            print(f"SKIP missing: {label} ({run_dir})", flush=True)
            continue
        t0 = time.time()
        mech_point = load_point(run_dir)
        results, n_common = compare(label, mech_point, curve_points, sims, user_ids, RNG)
        elapsed = time.time() - t0
        if results is None:
            print(f"SKIP (too few common users={n_common}): {label}", flush=True)
            continue
        for metric, r in results.items():
            rows.append(dict(dataset=ds, comparison=label, metric=metric, n_common_users=n_common,
                              **r))
        print(f"[{ds}] {label}: done in {elapsed:.1f}s, n_common_users={n_common}", flush=True)


# ---------------- GeoLife ----------------
geo_curve = geolife_curve_points()
geo_mechs = []
for l4 in (0.3, 1.0, 3.0):
    geo_mechs.append((f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"geolife_ed_soft_lambda{l4}_10seed"))
for l4 in (0.3, 1.0, 3.0):
    geo_mechs.append((f"ed_soft_location_gradmatched_lambda{l4}", PAPER_BASE / f"geolife_ed_soft_location_gradmatched_lambda{l4}_3seed"))
for basis in ("entropy", "location_distribution"):
    for l4 in (0.3, 1.0, 3.0):
        geo_mechs.append((f"lipschitz_{basis}_lambda{l4}", TASK2_BASE / f"geolife_lipschitz_{basis}_lambda{l4}_10seed"))
for l4 in (0.3, 1.0):
    geo_mechs.append((f"group_dro_lambda{l4}", TASK2_BASE / f"geolife_group_dro_lambda{l4}_3seed"))
for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
    geo_mechs.append((f"gi_dp_eps{eps}", GI_DP_BASE / f"geolife_gi_dp_eps{eps}"))
run_dataset("geolife", geo_curve, geo_mechs)

# base table: no-privacy -> plain lambda3=0.3
base_point = load_point(RETRAIN_BASE / "geolife_no_privacy")
user_ids, sims = load_sims("geolife")
base_curve_point = [p for p in geo_curve if p["source"] == "geolife_plain_lambda0.0_10seed"]
results, n_common = compare("base_table_no_privacy_vs_plain0.3", base_point, base_curve_point, sims, user_ids, RNG)
if results:
    for metric, r in results.items():
        rows.append(dict(dataset="geolife", comparison="base_table_no_privacy_vs_plain0.3",
                          metric=metric, n_common_users=n_common, **r))
print("[geolife] base table done", flush=True)

# ---------------- Foursquare ----------------
fsq_curve = std_curve_points("foursquare")
fsq_mechs = []
for l4 in (0.3, 1.0, 3.0):
    fsq_mechs.append((f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"foursquare_ed_soft_lambda{l4}"))
for basis in ("entropy", "location_distribution"):
    for l4 in (0.3, 1.0, 3.0):
        fsq_mechs.append((f"lipschitz_{basis}_lambda{l4}", TASK2_BASE / f"foursquare_lipschitz_{basis}_lambda{l4}_3seed"))
for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
    fsq_mechs.append((f"gi_dp_eps{eps}", GI_DP_BASE / f"foursquare_gi_dp_eps{eps}"))
run_dataset("foursquare", fsq_curve, fsq_mechs)

# ---------------- Gowalla-SF-Bay ----------------
gow_curve = std_curve_points("gowalla_sf_bay")
gow_mechs = []
for l4 in (0.3, 1.0, 3.0):
    gow_mechs.append((f"ed_soft_entropy_lambda{l4}", RETRAIN_BASE / f"gowalla_sf_bay_ed_soft_lambda{l4}"))
for basis in ("entropy", "location_distribution"):
    for l4 in (0.3, 1.0, 3.0):
        gow_mechs.append((f"lipschitz_{basis}_lambda{l4}", TASK2_BASE / f"gowalla_sf_bay_lipschitz_{basis}_lambda{l4}_3seed"))
for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
    gow_mechs.append((f"gi_dp_eps{eps}", GI_DP_BASE / f"gowalla_sf_bay_gi_dp_eps{eps}"))
run_dataset("gowalla_sf_bay", gow_curve, gow_mechs)

out_df = pd.DataFrame(rows)
out_path = "reports/figures/paper/item2_bootstrap_cis_matched_utility.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}, {len(out_df)} rows")
