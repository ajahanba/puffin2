#!/usr/bin/env python
"""Checkpoint 04c: re-analyze Phase 4's EXISTING runs under the PI's new
primary standard (the naive/hardened external attacker -- NOT the retired
adaptive attacker), using data already on disk. No retraining.

For each (dataset, formulation), builds two Pareto fronts across the
lambda4 in {0.3, 1.0, 3.0} points already run:
  1. utility vs. the PRE-SPECIFIED lowest-entropy decile's (decile 0)
     mean post_top1 risk.
  2. utility vs. "max-min" risk: the worst-off decile's risk, where the
     worst-off decile is CHOSEN on a held-out half of the attacker seeds
     (0,1,2) and MEASURED on the other half (3,4) -- not picked by
     peeking at the same seeds used to report the number, per the PI's
     explicit instruction.
Plain Mo-PAE (lambda4=0, same lambda1-3) is the reference point/curve,
reused from Checkpoint 3b's existing data (not rerun).

Also runs a leveling-down check (puffin.metrics.leveling_down) against
plain Mo-PAE at matched utility (+/-0.02 top-1), by decile, with
seed-paired bootstrap CIs on the per-decile helped/harmed deltas. The
per-user minimum (>=10 test windows) filter is applied throughout.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.metrics.privacy import privacy_gain
from puffin.metrics.leveling_down import leveling_down_check
from puffin.attacks.analysis import bootstrap_decile_ci

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
FORMULATIONS = ["ed_soft", "lipschitz", "floor", "variance", "group_dro", "per_user_minimax", "group_adversarial"]
LAMBDAS = [0.3, 1.0, 3.0]
MIN_TEST_WINDOWS = 10
UTILITY_MATCH_TOL = 0.02


def _load_run(dataset, formulation, lambda4):
    """Returns (utility, per_user_df) or None if missing. ed_soft reuses
    Checkpoint 3b's data (it has the full lambda grid already); everything
    else reuses Checkpoint 4's phase4/equalize grid."""
    if formulation == "ed_soft":
        p = pathlib.Path(f"reports/figures/phase3b/{dataset}_lambda4_{lambda4}/summary.json")
        csv_p = pathlib.Path(f"reports/figures/phase3b/{dataset}_lambda4_{lambda4}/per_user_3b.csv")
        if not p.exists():
            return None
        df = pd.read_csv(csv_p)
        # phase3b doesn't save utility -- flagged as a gap, see main()
        return None, df
    suffix = "" if lambda4 == 1.0 else f"_lambda{lambda4}"
    p = pathlib.Path(f"reports/figures/phase4/{dataset}_{formulation}{suffix}/summary.json")
    csv_p = pathlib.Path(f"reports/figures/phase4/{dataset}_{formulation}{suffix}/per_user_phase4.csv")
    if not p.exists():
        return None
    summ = json.load(open(p))
    df = pd.read_csv(csv_p)
    return summ["utility_next_location_top1"], df


def _apply_min_filter(df):
    return df[df["n_windows"] >= MIN_TEST_WINDOWS].copy()


def _held_out_worst_decile(df, select_seeds=(0, 1, 2), measure_seeds=(3, 4)):
    """Picks the worst-off decile using select_seeds' mean post_top1 by
    decile, then returns that decile's mean post_top1 on measure_seeds --
    a genuinely held-out max-min risk, not picked by peeking at the same
    data reported."""
    select_cols = [f"post_top1_seed{s}" for s in select_seeds if f"post_top1_seed{s}" in df.columns]
    measure_cols = [f"post_top1_seed{s}" for s in measure_seeds if f"post_top1_seed{s}" in df.columns]
    if not select_cols or not measure_cols:
        return None, None
    df = df.copy()
    df["_select_mean"] = df[select_cols].mean(axis=1)
    df["_measure_mean"] = df[measure_cols].mean(axis=1)
    by_decile_select = df.groupby("entropy_decile")["_select_mean"].mean()
    worst_decile = int(by_decile_select.idxmax())
    measured = float(df.loc[df["entropy_decile"] == worst_decile, "_measure_mean"].mean())
    return worst_decile, measured


def main():
    pareto_data = {}
    leveling_down_results = []
    gaps = []

    for ds in DATASETS:
        pareto_data[ds] = {}
        # plain Mo-PAE reference (lambda4=0, Checkpoint 3b)
        plain_p = pathlib.Path(f"reports/figures/phase3b/{ds}_lambda4_0.0/per_user_3b.csv")
        plain_df = _apply_min_filter(pd.read_csv(plain_p)) if plain_p.exists() else None
        plain_summary_p = pathlib.Path(f"reports/figures/phase3b/{ds}_lambda4_0.0/summary.json")
        plain_utility = None  # phase3b never computed utility -- see gaps below
        if plain_df is None:
            gaps.append(f"{ds}: no plain Mo-PAE (lambda4=0) reference data found")
        else:
            gaps.append(f"{ds}: plain Mo-PAE reference has no recorded utility (phase3b never computed it) "
                         f"-- Pareto fronts below use formulations' own utility axis only; plain Mo-PAE is "
                         f"plotted as a horizontal reference line on the risk axis, not matched on utility.")

        for f in FORMULATIONS:
            points = []
            for l4 in LAMBDAS:
                loaded = _load_run(ds, f, l4)
                if loaded is None:
                    gaps.append(f"{ds}/{f}/lambda4={l4}: MISSING run")
                    continue
                utility, df = loaded
                if utility is None:
                    gaps.append(f"{ds}/{f}/lambda4={l4}: no utility recorded (ed_soft/phase3b source) "
                                f"-- excluded from Pareto front, would need retraining with utility logging "
                                f"to include.")
                    continue
                df = _apply_min_filter(df)
                decile0 = df[df["entropy_decile"] == 0]
                risk_d0 = float(decile0["post_top1"].mean()) if len(decile0) else None
                worst_decile, risk_maxmin = _held_out_worst_decile(df)
                points.append(dict(lambda4=l4, utility=utility, risk_decile0=risk_d0,
                                    worst_decile=worst_decile, risk_maxmin=risk_maxmin, n_users=len(df)))

                # leveling-down vs plain Mo-PAE at matched utility
                if plain_df is not None and plain_utility is not None and abs(utility - plain_utility) <= UTILITY_MATCH_TOL:
                    merged = df.merge(plain_df[["user_id", "raw_top1", "post_top1", "entropy_decile"]],
                                       on="user_id", suffixes=("", "_plain"))
                    merged["gain_mech"] = privacy_gain(merged["raw_top1"].to_numpy(), merged["post_top1"].to_numpy())
                    merged["gain_plain"] = privacy_gain(merged["raw_top1"].to_numpy(),
                                                         merged["post_top1_plain"].to_numpy())
                    for dec in sorted(merged["entropy_decile"].unique()):
                        sub = merged[merged["entropy_decile"] == dec]
                        if len(sub) < 2:
                            continue
                        gm = pd.Series(sub["gain_mech"].to_numpy(), index=sub["user_id"])
                        gp = pd.Series(sub["gain_plain"].to_numpy(), index=sub["user_id"])
                        ld = leveling_down_check(gm, gp)
                        ld.update(dataset=ds, formulation=f, lambda4=l4, decile=int(dec))
                        leveling_down_results.append(ld)
            if points:
                pareto_data[ds][f] = points
            else:
                gaps.append(f"{ds}/{f}: NO usable points at all for the Pareto front")

    pathlib.Path("reports/figures/phase4c").mkdir(parents=True, exist_ok=True)
    json.dump(pareto_data, open("reports/figures/phase4c/pareto_fronts.json", "w"), indent=2)
    json.dump(leveling_down_results, open("reports/figures/phase4c/leveling_down.json", "w"), indent=2)
    json.dump(gaps, open("reports/figures/phase4c/gaps.json", "w"), indent=2)

    print(f"Pareto fronts built for {sum(len(v) for v in pareto_data.values())} (dataset,formulation) pairs")
    print(f"Leveling-down comparisons: {len(leveling_down_results)}")
    print(f"Gaps/issues found: {len(gaps)}")
    for g in gaps:
        print(" -", g)


if __name__ == "__main__":
    main()
