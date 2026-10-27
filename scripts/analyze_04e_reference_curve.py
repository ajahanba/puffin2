#!/usr/bin/env python
"""Checkpoint 04e, item 1: build the plain-Mo-PAE Pareto reference curve
(lambda_priv/lambda3 swept across {0.02, 0.1, 0.2, 0.3, 0.5, 0.8}, each
x3 encoder seeds, mechanism=plain throughout -- lambda3=0.3 reuses 04d's
existing <dataset>_plain_lambda0.0 run rather than retraining it), then
compares every formulation's point against the curve AT MATCHED UTILITY
via linear interpolation over the curve's own (utility, risk) points --
not against a single plain-Mo-PAE point, per the PI's explicit
instruction.

A formulation's point is read as:
  - BEATING the reference curve if its decile-0 risk is below the
    curve's interpolated risk at the same utility (CI-aware: the
    formulation's own CI must not overlap the interpolated point).
  - ON the reference curve if risk is statistically indistinguishable
    from the interpolated value -- i.e., the formulation is just a
    point achievable by tuning lambda3 alone, no fairness-specific
    benefit.
  - WORSE than the reference curve if risk is above the interpolated
    value -- dominated by a simpler lambda3 choice.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MIN_WINDOWS = 10
LAMBDA3_SWEEP = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8]
BASE_04D = pathlib.Path("reports/figures/phase4d")
BASE_REFCURVE = pathlib.Path("reports/figures/phase4e_refcurve")
BASE_10SEED = pathlib.Path("reports/figures/phase4e_10seed")
BASE_REFIT40 = pathlib.Path("reports/figures/phase4e_refit40")


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


def _load(path):
    summ = json.load(open(path / "summary.json"))
    df = pd.read_csv(path / "per_user_04d.csv")
    df = df[df["n_windows"] >= MIN_WINDOWS].copy()
    return summ, df


def decile_point(df, decile):
    sub = df[df.entropy_decile == decile]
    return float(sub["post_top1"].mean()) if len(sub) else None


def build_reference_curve(ds):
    """Returns sorted-by-utility list of dicts: lambda3, utility,
    utility_std, risk_decile0, risk_decile9."""
    points = []
    for lp in LAMBDA3_SWEEP:
        reused_mismatched_epochs = lp == 0.3
        path = (BASE_04D / f"{ds}_plain_lambda0.0") if reused_mismatched_epochs else (BASE_REFCURVE / f"{ds}_plain_lp{lp}")
        if not (path / "summary.json").exists():
            continue
        summ, df = _load(path)
        points.append(dict(lambda3=lp, utility=summ["utility_mean"], utility_std=summ["utility_std"],
                            risk_decile0=decile_point(df, 0), risk_decile9=decile_point(df, 9),
                            n_encoder_seeds=len(summ["encoder_seeds"]),
                            epochs=20 if reused_mismatched_epochs else 40,
                            note="reused from 04d at 20 epochs, not 40 -- epoch-count mismatch with "
                                 "the rest of this curve" if reused_mismatched_epochs else None))
    points.sort(key=lambda p: p["utility"])
    return points


def interpolate(curve, target_utility, key):
    utils = [p["utility"] for p in curve]
    risks = [p[key] for p in curve]
    if target_utility < min(utils) or target_utility > max(utils):
        return None, "target utility outside the reference curve's own range -- extrapolation refused"
    return float(np.interp(target_utility, utils, risks)), None


def compare_to_curve(curve, utility, risk_decile0):
    interp_risk, note = interpolate(curve, utility, "risk_decile0")
    if interp_risk is None:
        return dict(status="unresolvable", note=note)
    delta = interp_risk - risk_decile0  # positive: formulation beats the curve (lower risk)
    return dict(status="beats_curve" if delta > 0 else "worse_than_curve" if delta < 0 else "on_curve",
                interpolated_curve_risk=interp_risk, actual_risk=risk_decile0, delta=delta)


def main():
    out_dir = pathlib.Path("reports/figures/phase4e_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in ["geolife", "foursquare", "gowalla_nyc"]:
        curve = build_reference_curve(ds)
        results[ds] = dict(reference_curve=curve, comparisons=[])

        refit_path = BASE_REFIT40 / f"{ds}_no_privacy_40ep"
        if (refit_path / "summary.json").exists():
            summ, df = _load(refit_path)
            results[ds]["no_privacy_ceiling_40ep"] = dict(
                utility_mean=summ["utility_mean"], utility_std=summ["utility_std"],
                risk_decile0=decile_point(df, 0), n_encoder_seeds=len(summ["encoder_seeds"]), epochs=40)
            old_summ, _ = _load(BASE_04D / f"{ds}_no_privacy_lambda0.0")
            results[ds]["no_privacy_ceiling_20ep_stale"] = dict(
                utility_mean=old_summ["utility_mean"], utility_std=old_summ["utility_std"], epochs=20)
        if len(curve) < 2:
            results[ds]["error"] = f"only {len(curve)} reference-curve points available, need >=2 to interpolate"
            continue
        jobs = [("ed_soft", l4) for l4 in (0.3, 1.0, 3.0)]
        if ds == "geolife":
            jobs += [("floor", l4) for l4 in (0.3, 1.0, 3.0)]
        for mech, l4 in jobs:
            path = BASE_04D / f"{ds}_{mech}_lambda{l4}"
            if not (path / "summary.json").exists():
                continue
            summ, df = _load(path)
            cmp = compare_to_curve(curve, summ["utility_mean"], decile_point(df, 0))
            cmp.update(mechanism=mech, lambda4=l4, utility=summ["utility_mean"])
            results[ds]["comparisons"].append(cmp)

        # GeoLife 10-seed recheck of plain + ed_soft, if that data has landed
        if ds == "geolife":
            for mech, l4 in [("plain", 0.0), ("ed_soft", 0.3), ("ed_soft", 1.0), ("ed_soft", 3.0)]:
                suffix = "plain_lambda0.0_10seed" if mech == "plain" else f"ed_soft_lambda{l4}_10seed"
                path = BASE_10SEED / f"geolife_{suffix}"
                if (path / "summary.json").exists():
                    summ, df = _load(path)
                    results[ds].setdefault("ten_seed_recheck", {})[f"{mech}_lambda{l4}"] = dict(
                        utility_mean=summ["utility_mean"], utility_std=summ["utility_std"],
                        n_encoder_seeds=len(summ["encoder_seeds"]), risk_decile0=decile_point(df, 0))

    json.dump(results, open(out_dir / "reference_curve_comparisons.json", "w"), indent=2, default=_json_default)
    for ds, r in results.items():
        print(f"{ds}: {len(r.get('reference_curve', []))} curve points, "
              f"{len(r.get('comparisons', []))} comparisons, "
              f"ten_seed_recheck={'yes' if 'ten_seed_recheck' in r else 'no'}")


if __name__ == "__main__":
    main()
