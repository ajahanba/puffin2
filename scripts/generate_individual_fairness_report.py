#!/usr/bin/env python
"""Generates reports/06_individual_fairness.md from Task 1's per-dataset
JSON outputs (reports/figures/individual_fairness/*_individual_fairness.json)
plus the raw cached similarity matrices (*_similarities.npz, *_user_ids.json),
per the PI's fairness-reframing instruction (2026-11-02) and the
delta=0.8/leading-with-top10pct correction.
"""
import json
import pathlib

import numpy as np
import pandas as pd

BASE = pathlib.Path("reports/figures/individual_fairness")
DATASETS = ["geolife", "foursquare", "gowalla_sf_bay"]
SIM_NAMES = ["entropy", "location_distribution", "trajectory_shape", "radius_of_gyration"]
SIM_LABELS = {"entropy": "Entropy", "location_distribution": "Location-distribution",
              "trajectory_shape": "Trajectory-shape", "radius_of_gyration": "Radius of gyration"}


def load_all():
    data = {}
    for ds in DATASETS:
        p = BASE / f"{ds}_individual_fairness.json"
        if p.exists():
            data[ds] = json.loads(p.read_text())
    return data


def total_pairs(n):
    return n * (n - 1) // 2


def delta_fraction_table(data):
    """% of ALL pairs counted similar at delta=0.8, per definition, per
    dataset -- using the no_privacy run's n_users as the reference pair
    count (any run intersects down slightly, close enough for this
    diagnostic table)."""
    rows = []
    for ds, runs in data.items():
        ref_run = next((r for r in runs if "no_privacy" in r), next(iter(runs)))
        n_users = runs[ref_run]["n_users"]
        tp = total_pairs(n_users)
        row = {"dataset": ds, "n_users": n_users, "total_pairs": tp}
        for sim in SIM_NAMES:
            n_pairs = runs[ref_run][sim]["delta_0_8"]["n_pairs"]
            row[sim] = n_pairs / tp if tp else None
        rows.append(row)
    return pd.DataFrame(rows)


def pair_agreement(ds):
    """Jaccard overlap between each pair of similarity definitions' top-10%
    sets, plus a concrete entropy-similar/location-dissimilar example."""
    sims = np.load(BASE / f"{ds}_similarities.npz")
    user_ids = json.loads((BASE / f"{ds}_user_ids.json").read_text())
    n = len(user_ids)
    iu = np.triu_indices(n, k=1)

    top_sets = {}
    for sim_name in SIM_NAMES:
        vals = sims[sim_name][iu]
        k = max(1, int(round(0.10 * len(vals))))
        thresh = np.partition(vals, -k)[-k]
        top_sets[sim_name] = set(np.where(vals >= thresh)[0])

    agreement = {}
    for i, a in enumerate(SIM_NAMES):
        for b in SIM_NAMES[i + 1:]:
            inter = len(top_sets[a] & top_sets[b])
            union = len(top_sets[a] | top_sets[b])
            agreement[f"{a}_vs_{b}"] = inter / union if union else None

    entropy_vals = sims["entropy"][iu]
    loc_vals = sims["location_distribution"][iu]
    # entropy-similar (top decile) but location-dissimilar (bottom half)
    mask = (entropy_vals >= np.partition(entropy_vals, -max(1, int(0.10 * len(entropy_vals))))[-max(1, int(0.10 * len(entropy_vals)))]) & \
           (loc_vals <= np.median(loc_vals))
    example = None
    if mask.any():
        idx = np.where(mask)[0]
        pick = idx[np.argmin(loc_vals[idx])]  # most location-dissimilar among entropy-similar
        ui, uj = iu[0][pick], iu[1][pick]
        example = dict(user_a=user_ids[ui], user_b=user_ids[uj],
                       entropy_sim=float(entropy_vals[pick]), location_sim=float(loc_vals[pick]))
    return agreement, example


def matched_utility_if_rate(data, ds, mechanism_prefix="ed_soft", lambda4_vals=(0.3, 1.0, 3.0)):
    """Builds the plain lambda3 curve's (utility, IF-rate) pairs for each
    similarity x selection method, interpolates to each ed_soft point's
    utility, and reports the delta. Mirrors analyze_geolife_frontier.py's
    extrapolation-refusal rule."""
    runs = data[ds]
    curve_runs = {name: r for name, r in runs.items() if "_plain_lp" in name or name.endswith("_plain_lambda0.0_10seed")}
    results = {}
    summ_cache = {}

    def utility_of(run_name):
        if run_name not in summ_cache:
            p = pathlib.Path(f"reports/figures/retraining_to_convergence/{run_name}/summary.json")
            summ_cache[run_name] = json.loads(p.read_text())["utility_mean"] if p.exists() else None
        return summ_cache[run_name]

    curve = []
    for name in curve_runs:
        u = utility_of(name)
        if u is not None:
            curve.append((u, name))
    curve.sort()
    if len(curve) < 2:
        return results

    for l4 in lambda4_vals:
        ed_name = next((n for n in runs if n.startswith(f"{ds}_{mechanism_prefix}_lambda{l4}")), None)
        if ed_name is None:
            continue
        ed_u = utility_of(ed_name)
        if ed_u is None or ed_u < curve[0][0] or ed_u > curve[-1][0]:
            results[l4] = dict(status="unresolvable (outside curve range)")
            continue
        per_sim = {}
        for sim in SIM_NAMES:
            utils = [c[0] for c in curve]
            rates_top = [runs[c[1]][sim]["top_10pct"]["rate"] for c in curve]
            interp_rate = float(np.interp(ed_u, utils, [r if r is not None else np.nan for r in rates_top]))
            ed_rate = runs[ed_name][sim]["top_10pct"]["rate"]
            per_sim[sim] = dict(plain_curve_interp=interp_rate, ed_soft=ed_rate,
                                 delta=None if ed_rate is None else ed_rate - interp_rate)
        results[l4] = dict(status="ok", utility=ed_u, per_similarity=per_sim)
    return results


def main():
    data = load_all()
    print("Loaded datasets:", list(data.keys()))
    delta_tab = delta_fraction_table(data)
    print(delta_tab.to_string())
    for ds in data:
        agreement, example = pair_agreement(ds)
        print(f"\n[{ds}] pairwise agreement (Jaccard, top-10%% sets):")
        for k, v in agreement.items():
            print(f"  {k}: {v:.3f}")
        print(f"  entropy-similar/location-dissimilar example: {example}")
        mr = matched_utility_if_rate(data, ds)
        print(f"  matched-utility PUFFIN-vs-curve IF-rate: {json.dumps(mr, indent=2, default=float)}")


if __name__ == "__main__":
    main()
