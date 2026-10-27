#!/usr/bin/env python
"""Checkpoint 04d: analyze the PI-approved batch (plain Mo-PAE, ed_soft
with utility logged, the no-privacy utility-ceiling reference, and
GeoLife's fixed floor_loss reruns), all under the new >=3-encoder-seed /
hardened-external-attacker standard. Three things, per the PI's explicit
asks:

1. Apply the corrected decile-reduction metrics (puffin.metrics.
   decile_reduction: decile_reduction_ci, residual_risk_disparity) to
   every run -- the headline statistic, not the old per-user ratio.
2. The in-model-adversary-vs-external-attacker confidence comparison by
   decile, for the "plain" mechanism on all 3 datasets -- the
   mechanistic illustration of why loss-based fairness can't target
   real risk (floor_loss's own near-chance, decile-flat confidence next
   to the external attacker's higher, decile-dependent accuracy on the
   SAME features).
3. Rebuild Checkpoint 04c's Pareto fronts with plain Mo-PAE and the
   no-privacy reference overlaid (now both have a recorded utility,
   closing 04c's blocking gap), using 04d's own ed_soft/floor data (3
   encoder seeds, properly logged utility) in place of the old
   single-seed/no-utility versions. Runs the leveling-down check
   (puffin.metrics.leveling_down) for every (mechanism, lambda4) point
   against plain Mo-PAE at matched utility (+/-0.02), by decile --
   computed wherever a match exists; the 5 single-encoder-seed Phase 4
   formulations (lipschitz/variance/group_dro/per_user_minimax/
   group_adversarial) are included but flagged with an explicit
   seed-count-mismatch caveat (1 encoder seed vs. plain's 3), since
   they were out of scope for this round's retraining.

No new training. Pure analysis of what's already on disk.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.metrics.decile_reduction import decile_reduction_ci, residual_risk_disparity
from puffin.metrics.leveling_down import leveling_down_check
from puffin.metrics.privacy import privacy_gain

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
OLD_FORMULATIONS = ["lipschitz", "variance", "group_dro", "per_user_minimax", "group_adversarial"]
MIN_WINDOWS = 10
UTILITY_MATCH_TOL = 0.02

BASE_04D = pathlib.Path("reports/figures/phase4d")
BASE_PHASE4 = pathlib.Path("reports/figures/phase4")


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


def load_04d_run(ds, mech, l4):
    d = BASE_04D / f"{ds}_{mech}_lambda{l4}"
    summ = json.load(open(d / "summary.json"))
    df = pd.read_csv(d / "per_user_04d.csv")
    df = df[df["n_windows"] >= MIN_WINDOWS].copy()
    return summ, df


def load_old_phase4_run(ds, formulation, l4):
    """Single-encoder-seed Phase 4 grid (unchanged, not part of this
    round's retraining). Returns (utility_or_None, df_or_None)."""
    suffix = "" if l4 == 1.0 else f"_lambda{l4}"
    p = BASE_PHASE4 / f"{ds}_{formulation}{suffix}"
    if not (p / "summary.json").exists():
        return None, None
    summ = json.load(open(p / "summary.json"))
    df = pd.read_csv(p / "per_user_phase4.csv")
    df = df[df["n_windows"] >= MIN_WINDOWS].copy()
    return summ.get("utility_next_location_top1"), df


def decile_stats(df):
    raw = df["raw_top1"].to_numpy()
    post = df["post_top1"].to_numpy()
    dec = df["entropy_decile"].to_numpy()
    return dict(
        reduction=decile_reduction_ci(raw, post, dec),
        residual_risk_disparity=residual_risk_disparity(post, dec),
    )


def main():
    out_dir = pathlib.Path("reports/figures/phase4d_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. decile-reduction metrics for every 04d run ----
    per_run_metrics = {}
    jobs_per_dataset = {}
    for ds in DATASETS:
        jobs = [("plain", 0.0), ("no_privacy", 0.0)] + [("ed_soft", l4) for l4 in (0.0, 0.3, 1.0, 3.0)]
        if ds == "geolife":
            jobs += [("floor", l4) for l4 in (0.3, 1.0, 3.0)]
        jobs_per_dataset[ds] = jobs
        per_run_metrics[ds] = {}
        for mech, l4 in jobs:
            summ, df = load_04d_run(ds, mech, l4)
            key = f"{mech}_lambda{l4}"
            has_inmodel = "inmodel_confidence" in df.columns
            entry = dict(
                mechanism=mech, lambda4=l4,
                utility_mean=summ["utility_mean"], utility_std=summ["utility_std"],
                utility_per_encoder_seed=summ["utility_per_encoder_seed"],
                n_users=len(df), chance_level=summ.get("chance_level"),
                has_inmodel_confidence=has_inmodel,
                **decile_stats(df),
            )
            if has_inmodel:
                entry["inmodel_confidence_decile0"] = float(df[df.entropy_decile == 0]["inmodel_confidence"].mean())
                entry["inmodel_confidence_decile9"] = float(df[df.entropy_decile == 9]["inmodel_confidence"].mean())
            per_run_metrics[ds][key] = entry
    json.dump(per_run_metrics, open(out_dir / "decile_reduction_metrics.json", "w"),
              indent=2, default=_json_default)

    # ---- 2. in-model vs. external attacker, by decile, plain mechanism ----
    inmodel_vs_external = {}
    for ds in DATASETS:
        summ, df = load_04d_run(ds, "plain", 0.0)
        by_dec = df.groupby("entropy_decile").agg(
            external_top1=("post_top1", "mean"),
            inmodel_confidence=("inmodel_confidence", "mean"),
            n_users=("user_id", "count"),
        )
        inmodel_vs_external[ds] = dict(
            chance_level=summ["chance_level"],
            by_decile={str(k): v for k, v in by_dec.to_dict(orient="index").items()},
        )
    json.dump(inmodel_vs_external, open(out_dir / "inmodel_vs_external.json", "w"),
              indent=2, default=_json_default)

    # ---- 3a. rebuilt Pareto fronts: 04d's own plain/no_privacy/ed_soft/floor
    #          + the 5 old single-seed formulations, all referenced against
    #          plain Mo-PAE and no-privacy from 04d ----
    pareto = {}
    for ds in DATASETS:
        pareto[ds] = {}
        plain_summ, plain_df = load_04d_run(ds, "plain", 0.0)
        nopriv_summ, nopriv_df = load_04d_run(ds, "no_privacy", 0.0)

        def point_from_04d(mech, l4):
            summ, df = load_04d_run(ds, mech, l4)
            d0 = df[df.entropy_decile == 0]
            worst_dec = int(df.groupby("entropy_decile")["post_top1"].mean().idxmax())
            d_worst = df[df.entropy_decile == worst_dec]
            return dict(lambda4=l4, utility=summ["utility_mean"], utility_std=summ["utility_std"],
                        risk_decile0=float(d0["post_top1"].mean()) if len(d0) else None,
                        worst_decile=worst_dec, risk_maxmin=float(d_worst["post_top1"].mean()),
                        n_users=len(df), n_encoder_seeds=len(summ["encoder_seeds"]))

        pareto[ds]["plain"] = [point_from_04d("plain", 0.0)]
        pareto[ds]["no_privacy"] = [point_from_04d("no_privacy", 0.0)]
        pareto[ds]["ed_soft"] = [point_from_04d("ed_soft", l4) for l4 in (0.0, 0.3, 1.0, 3.0)]
        if ds == "geolife":
            pareto[ds]["floor"] = [point_from_04d("floor", l4) for l4 in (0.3, 1.0, 3.0)]

        for f in OLD_FORMULATIONS:
            points = []
            for l4 in (0.3, 1.0, 3.0):
                utility, df = load_old_phase4_run(ds, f, l4)
                if df is None or utility is None:
                    continue
                d0 = df[df.entropy_decile == 0]
                worst_dec = int(df.groupby("entropy_decile")["post_top1"].mean().idxmax())
                d_worst = df[df.entropy_decile == worst_dec]
                points.append(dict(lambda4=l4, utility=utility,
                                    risk_decile0=float(d0["post_top1"].mean()) if len(d0) else None,
                                    worst_decile=worst_dec, risk_maxmin=float(d_worst["post_top1"].mean()),
                                    n_users=len(df), n_encoder_seeds=1,
                                    caveat="single encoder seed (not part of this round's retraining); "
                                           "plain/no-privacy reference points below are 3-encoder-seed "
                                           "04d data -- compared across a seed-count mismatch."))
            if points:
                pareto[ds][f] = points
    json.dump(pareto, open(out_dir / "pareto_fronts_rebuilt.json", "w"), indent=2, default=_json_default)

    # ---- 3b. leveling-down check vs. plain Mo-PAE at matched utility ----
    leveling_results = []
    for ds in DATASETS:
        plain_summ, plain_df = load_04d_run(ds, "plain", 0.0)
        plain_gain = pd.Series(
            privacy_gain(plain_df["raw_top1"].to_numpy(), plain_df["post_top1"].to_numpy()),
            index=plain_df["user_id"],
        )
        plain_decile = plain_df.set_index("user_id")["entropy_decile"]

        def run_leveling(mech, l4, utility, df, n_encoder_seeds, caveat=None):
            if abs(utility - plain_summ["utility_mean"]) > UTILITY_MATCH_TOL:
                leveling_results.append(dict(
                    dataset=ds, mechanism=mech, lambda4=l4, skipped=True,
                    reason=f"utility not matched within +/-{UTILITY_MATCH_TOL} "
                           f"(mech={utility:.4f}, plain={plain_summ['utility_mean']:.4f})",
                ))
                return
            gain_mech = pd.Series(privacy_gain(df["raw_top1"].to_numpy(), df["post_top1"].to_numpy()),
                                   index=df["user_id"])
            dec_mech = df.set_index("user_id")["entropy_decile"]
            for dec in sorted(df["entropy_decile"].unique()):
                users = dec_mech[dec_mech == dec].index
                gm = gain_mech.reindex(users).dropna()
                gp = plain_gain.reindex(plain_decile[plain_decile == dec].index).dropna()
                common = gm.index.intersection(gp.index)
                gm_c, gp_c = gm.loc[common], gp.loc[common]
                if len(gm_c) < 2:
                    continue
                ld = leveling_down_check(gm_c, gp_c)
                ld.update(dataset=ds, mechanism=mech, lambda4=l4, decile=int(dec),
                          n_encoder_seeds=n_encoder_seeds, caveat=caveat)
                leveling_results.append(ld)

        for l4 in (0.3, 1.0, 3.0):
            summ, df = load_04d_run(ds, "ed_soft", l4)
            run_leveling("ed_soft", l4, summ["utility_mean"], df, len(summ["encoder_seeds"]))
        if ds == "geolife":
            for l4 in (0.3, 1.0, 3.0):
                summ, df = load_04d_run(ds, "floor", l4)
                run_leveling("floor", l4, summ["utility_mean"], df, len(summ["encoder_seeds"]))
        for f in OLD_FORMULATIONS:
            for l4 in (0.3, 1.0, 3.0):
                utility, df = load_old_phase4_run(ds, f, l4)
                if df is None or utility is None:
                    continue
                run_leveling(f, l4, utility, df, 1,
                             caveat="single encoder seed vs. plain's 3 -- seed-count mismatch, "
                                    "not a like-for-like comparison")

    json.dump(leveling_results, open(out_dir / "leveling_down_rebuilt.json", "w"),
              indent=2, default=_json_default)

    print("Wrote:")
    for f in ["decile_reduction_metrics.json", "inmodel_vs_external.json",
              "pareto_fronts_rebuilt.json", "leveling_down_rebuilt.json"]:
        print(" -", out_dir / f)
    print(f"\n{sum(len(v) for v in per_run_metrics.values())} runs analyzed for decile-reduction metrics")
    print(f"{sum(len(v) for v in pareto.values())} (dataset,formulation) Pareto fronts")
    n_skip = sum(1 for r in leveling_results if r.get("skipped"))
    print(f"{len(leveling_results) - n_skip} leveling-down decile comparisons computed, {n_skip} skipped (utility not matched)")


if __name__ == "__main__":
    main()
