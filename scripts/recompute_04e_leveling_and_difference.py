#!/usr/bin/env python
"""Checkpoint 04e, items 3-5 (the parts that don't need new training):
recompute the leveling-down check on 04d's existing data with the
recalibrated metric (absolute post-risk, seed-to-seed null, tested
against that null), and add the absolute risk difference (decile 0 -
decile 9) alongside the existing RRD ratio for every 04d run.

The seed-to-seed null is built from PLAIN Mo-PAE's own 3 encoder-seed
replicates (post_top1_encseed0/1/2 columns already in
reports/figures/phase4d/<dataset>_plain_lambda0.0/per_user_04d.csv) --
this is the "plain vs plain across encoder seeds" null the PI asked
for. It is necessarily thin (3 seeds -> 6 ordered pairs per user) for
every dataset except GeoLife, where the 10-encoder-seed rerun (queued
separately, reports/figures/phase4e_10seed/) will give a much richer
null (90 ordered pairs per user) once it lands -- this script uses the
3-seed null throughout for now and is re-run once the 10-seed data is
available, per the report's own stated limitation.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.metrics.decile_reduction import residual_risk_disparity, residual_risk_difference
from puffin.metrics.leveling_down import leveling_down_check, seed_to_seed_null

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
MIN_WINDOWS = 10
UTILITY_MATCH_TOL = 0.02
BASE = pathlib.Path("reports/figures/phase4d")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def load_run(ds, mech, l4):
    d = BASE / f"{ds}_{mech}_lambda{l4}"
    summ = json.load(open(d / "summary.json"))
    df = pd.read_csv(d / "per_user_04d.csv")
    df = df[df["n_windows"] >= MIN_WINDOWS].copy()
    return summ, df


def main():
    out_dir = pathlib.Path("reports/figures/phase4e_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- item 4: residual_risk_difference alongside RRD ratio, every run ---
    difference_results = {}
    for ds in DATASETS:
        jobs = [("plain", 0.0), ("no_privacy", 0.0)] + [("ed_soft", l4) for l4 in (0.0, 0.3, 1.0, 3.0)]
        if ds == "geolife":
            jobs += [("floor", l4) for l4 in (0.3, 1.0, 3.0)]
        difference_results[ds] = {}
        for mech, l4 in jobs:
            summ, df = load_run(ds, mech, l4)
            post = df["post_top1"].to_numpy()
            dec = df["entropy_decile"].to_numpy()
            ratio = residual_risk_disparity(post, dec)
            diff = residual_risk_difference(post, dec)
            difference_results[ds][f"{mech}_lambda{l4}"] = dict(
                utility_mean=summ["utility_mean"], utility_std=summ["utility_std"],
                decile0_risk=float(df[df.entropy_decile == 0]["post_top1"].mean()),
                decile9_risk=float(df[df.entropy_decile == 9]["post_top1"].mean()),
                rrd_ratio=ratio, rrd_difference=diff,
            )
    json.dump(difference_results, open(out_dir / "disparity_ratio_and_difference.json", "w"),
              indent=2, default=_json_default)

    # --- item 3: recalibrated leveling-down check, every ed_soft/floor
    #     point vs plain Mo-PAE, using plain's own 3-seed null ---
    leveling_results = []
    for ds in DATASETS:
        plain_summ, plain_df = load_run(ds, "plain", 0.0)
        plain_seed_cols_full = [plain_df[f"post_top1_encseed{s}"] for s in (0, 1, 2)]
        plain_post_by_user = plain_df.set_index("user_id")["post_top1"]
        plain_decile = plain_df.set_index("user_id")["entropy_decile"]

        jobs = [("ed_soft", l4) for l4 in (0.3, 1.0, 3.0)]
        if ds == "geolife":
            jobs += [("floor", l4) for l4 in (0.3, 1.0, 3.0)]

        for mech, l4 in jobs:
            summ, df = load_run(ds, mech, l4)
            if abs(summ["utility_mean"] - plain_summ["utility_mean"]) > UTILITY_MATCH_TOL:
                leveling_results.append(dict(dataset=ds, mechanism=mech, lambda4=l4, skipped=True,
                                              reason=f"utility not matched within +/-{UTILITY_MATCH_TOL} "
                                                     f"(mech={summ['utility_mean']:.4f}, "
                                                     f"plain={plain_summ['utility_mean']:.4f})"))
                continue
            mech_post_by_user = df.set_index("user_id")["post_top1"]
            dec_mech = df.set_index("user_id")["entropy_decile"]
            for dec in sorted(df["entropy_decile"].unique()):
                users_mech = dec_mech[dec_mech == dec].index
                users_plain = plain_decile[plain_decile == dec].index
                common = users_mech.intersection(users_plain)
                if len(common) < 2:
                    continue
                null_mask = plain_df["entropy_decile"] == dec
                null = seed_to_seed_null([c[null_mask].to_numpy() for c in plain_seed_cols_full])
                ld = leveling_down_check(mech_post_by_user.loc[common], plain_post_by_user.loc[common], null)
                ld.update(dataset=ds, mechanism=mech, lambda4=l4, decile=int(dec), null_source="plain_3seed")
                leveling_results.append(ld)

    json.dump(leveling_results, open(out_dir / "leveling_down_recalibrated.json", "w"),
              indent=2, default=_json_default)

    n_skip = sum(1 for r in leveling_results if r.get("skipped"))
    n_flagged = sum(1 for r in leveling_results if r.get("leveling_down_pattern"))
    print(f"Wrote {out_dir / 'disparity_ratio_and_difference.json'}")
    print(f"Wrote {out_dir / 'leveling_down_recalibrated.json'}")
    print(f"{len(leveling_results) - n_skip} comparisons computed, {n_skip} skipped (utility not matched)")
    print(f"{n_flagged} of those computed comparisons flag leveling_down_pattern=True "
          f"(harmed rate beyond the seed-to-seed null AND harmed users were already better-protected)")


if __name__ == "__main__":
    main()
