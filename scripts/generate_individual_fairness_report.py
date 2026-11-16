#!/usr/bin/env python
"""Generates reports/06_individual_fairness.md from compute_individual_
fairness.py's per-dataset JSON outputs
(reports/figures/individual_fairness/*_individual_fairness.json) plus the
raw cached similarity matrices (*_similarities.npz, *_user_ids.json).
"""
import json
import pathlib

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")
import scripts.compute_individual_fairness as cif
import puffin.metrics.individual_fairness as ifm

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


def load_similarities(ds):
    """Loads the cached similarity matrices, either as one combined .npz
    (GeoLife, Foursquare) or split per-definition .npz files (Gowalla-SF-Bay,
    whose combined float64 version was 250MB -- over GitHub's 100MB limit;
    split + float32 brings each under 30MB)."""
    combined = BASE / f"{ds}_similarities.npz"
    if combined.exists():
        return np.load(combined)
    return {name: np.load(BASE / f"{ds}_similarities_{name}.npz")[name] for name in SIM_NAMES}


def pair_agreement(ds):
    """Jaccard overlap between each pair of similarity definitions' top-10%
    sets, plus a concrete entropy-similar/location-dissimilar example."""
    sims = load_similarities(ds)
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


def no_privacy_vs_plain(data, ds):
    """Q1: does plain Mo-PAE violate individual fairness compared with
    no-privacy -- same user, same pairs, top-10% selection, per
    similarity definition."""
    runs = data[ds]
    if "{}_no_privacy".format(ds) not in runs:
        return None
    np_run = "{}_no_privacy".format(ds)
    plain_run = next((n for n in runs if "_plain_lp0.3" in n or n.endswith("_plain_lambda0.0_10seed")), None)
    if plain_run is None:
        return None
    out = {}
    for sim in SIM_NAMES:
        np_rate = runs[np_run][sim]["top_10pct"]["rate"]
        plain_rate = runs[plain_run][sim]["top_10pct"]["rate"]
        out[sim] = dict(no_privacy=np_rate, plain=plain_rate,
                         delta=None if (np_rate is None or plain_rate is None) else plain_rate - np_rate)
    return dict(no_privacy_run=np_run, plain_run=plain_run, per_similarity=out)


def compute_scale_free_table(data):
    """For every run in every dataset, under every similarity definition's
    top-10% selection: the scale-free IF ratio (mean |risk_i-risk_j| over
    top-10%-similar pairs / mean |risk_i-risk_j| over ALL pairs) with
    bootstrap CIs over seeds+pairs, plus mean risk. The scale-free ratio
    is reported because the raw absolute IF-rate mostly tracks overall
    privacy level, not genuine individual fairness."""
    out = {}
    for ds, runs in data.items():
        sims = load_similarities(ds)
        user_ids = json.loads((BASE / f"{ds}_user_ids.json").read_text())
        idx_map = {u: i for i, u in enumerate(user_ids)}
        run_dirs = {d.name: d for d in cif._run_dirs_for(ds)}
        ds_out = {}
        for run_name in runs:
            run_dir = run_dirs.get(run_name)
            if run_dir is None:
                continue
            valid_users, risk = cif.load_risk(run_dir, user_ids)
            if risk is None or len(valid_users) < 20:
                continue
            keep_idx = np.array([idx_map[u] for u in valid_users])
            per_sim = {}
            for sim_name in SIM_NAMES:
                sim = sims[sim_name][np.ix_(keep_idx, keep_idx)]
                i_top, j_top = ifm.select_similar_pairs(sim, top_frac=0.10)
                per_sim[sim_name] = ifm.scale_free_if(risk, i_top, j_top, n_boot=300)
            ds_out[run_name] = per_sim
        out[ds] = ds_out
    return out


def matched_utility_scale_free(data, scale_free, ds, mechanism_prefix="ed_soft", lambda4_vals=(0.3, 1.0, 3.0)):
    """Same matched-utility interpolation as matched_utility_if_rate, but
    comparing the scale-free ratio instead of the raw top-10% rate."""
    runs = data[ds]
    curve_runs = {name: r for name, r in runs.items() if "_plain_lp" in name or name.endswith("_plain_lambda0.0_10seed")}
    results = {}
    summ_cache = {}

    def utility_of(run_name):
        if run_name not in summ_cache:
            p = pathlib.Path(f"reports/figures/retraining_to_convergence/{run_name}/summary.json")
            summ_cache[run_name] = json.loads(p.read_text())["utility_mean"] if p.exists() else None
        return summ_cache[run_name]

    curve = [(utility_of(n), n) for n in curve_runs if utility_of(n) is not None]
    curve.sort()
    if len(curve) < 2:
        return results

    for l4 in lambda4_vals:
        ed_name = next((n for n in runs if n.startswith(f"{ds}_{mechanism_prefix}_lambda{l4}")), None)
        if ed_name is None or ed_name not in scale_free.get(ds, {}):
            continue
        ed_u = utility_of(ed_name)
        if ed_u is None or ed_u < curve[0][0] or ed_u > curve[-1][0]:
            results[l4] = dict(status="unresolvable (outside curve range)")
            continue
        per_sim = {}
        for sim in SIM_NAMES:
            utils = [c[0] for c in curve]
            ratios = [scale_free[ds][c[1]][sim]["ratio"] for c in curve if c[1] in scale_free[ds]]
            if len(ratios) != len(utils):
                continue
            interp_ratio = float(np.interp(ed_u, utils, ratios))
            ed_ratio = scale_free[ds][ed_name][sim]["ratio"]
            per_sim[sim] = dict(plain_curve_interp=interp_ratio, ed_soft=ed_ratio,
                                 delta=None if ed_ratio is None else ed_ratio - interp_ratio)
        results[l4] = dict(status="ok", utility=ed_u, per_similarity=per_sim)
    return results


def no_privacy_vs_plain_scale_free(scale_free, ds):
    """The same no-privacy-vs-plain comparison redone with the scale-free
    measure: the raw top-10% rate mostly tracks overall privacy level
    (mean risk), not genuine individual fairness."""
    runs = scale_free.get(ds, {})
    np_run = next((n for n in runs if n.endswith("_no_privacy")), None)
    plain_run = next((n for n in runs if "_plain_lp0.3" in n or n.endswith("_plain_lambda0.0_10seed")), None)
    if np_run is None or plain_run is None:
        return None
    out = {}
    for sim in SIM_NAMES:
        npv = runs[np_run][sim]
        pv = runs[plain_run][sim]
        out[sim] = dict(no_privacy_ratio=npv["ratio"], no_privacy_mean_risk=npv["mean_risk"],
                         plain_ratio=pv["ratio"], plain_mean_risk=pv["mean_risk"],
                         delta_ratio=None if (npv["ratio"] is None or pv["ratio"] is None) else pv["ratio"] - npv["ratio"])
    return dict(no_privacy_run=np_run, plain_run=plain_run, per_similarity=out)


def matched_mean_risk_comparison(data, scale_free, ds, mechanism_prefix="ed_soft", lambda4_vals=(0.3, 1.0, 3.0)):
    """Interpolates the plain lambda3 curve to each PUFFIN point's own MEAN RISK (not utility) and
    compares utility and decile-0 risk at that matched mean-risk level --
    the complementary question to the matched-utility comparison: at the
    same overall privacy level, which approach gives better utility and
    better worst-decile protection?"""
    runs = data[ds]
    curve_runs = {name: r for name, r in runs.items() if "_plain_lp" in name or name.endswith("_plain_lambda0.0_10seed")}
    results = {}

    def mean_risk_of(run_name, sim="entropy"):
        ds_sf = scale_free.get(ds, {})
        return ds_sf[run_name][sim].get("mean_risk") if run_name in ds_sf and sim in ds_sf[run_name] else None

    def utility_of(run_name):
        p = pathlib.Path(f"reports/figures/retraining_to_convergence/{run_name}/summary.json")
        return json.loads(p.read_text())["utility_mean"] if p.exists() else None

    def decile0_of(run_name):
        p = pathlib.Path(f"reports/figures/retraining_to_convergence/{run_name}/per_user_04d.csv")
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df = df[df.n_windows >= 10]
        return float(df[df.entropy_decile == 0]["post_top1"].mean())

    curve = []
    for name in curve_runs:
        mr = mean_risk_of(name)
        u = utility_of(name)
        d0 = decile0_of(name)
        if mr is not None and u is not None and d0 is not None:
            curve.append((mr, u, d0, name))
    curve.sort()
    if len(curve) < 2:
        return results

    for l4 in lambda4_vals:
        ed_name = next((n for n in runs if n.startswith(f"{ds}_{mechanism_prefix}_lambda{l4}")), None)
        if ed_name is None:
            continue
        ed_mr = mean_risk_of(ed_name)
        if ed_mr is None or ed_mr < curve[0][0] or ed_mr > curve[-1][0]:
            results[l4] = dict(status="unresolvable (outside curve range)",
                                ed_soft_mean_risk=ed_mr)
            continue
        mrs = [c[0] for c in curve]
        interp_u = float(np.interp(ed_mr, mrs, [c[1] for c in curve]))
        interp_d0 = float(np.interp(ed_mr, mrs, [c[2] for c in curve]))
        ed_u = utility_of(ed_name)
        ed_d0 = decile0_of(ed_name)
        results[l4] = dict(status="ok", matched_mean_risk=ed_mr,
                            plain_curve_interp=dict(utility=interp_u, decile0_risk=interp_d0),
                            ed_soft=dict(utility=ed_u, decile0_risk=ed_d0),
                            delta=dict(utility=ed_u - interp_u, decile0_risk=ed_d0 - interp_d0))
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
        npvp = no_privacy_vs_plain(data, ds)
        print(f"  no_privacy vs plain (top10%% IF-rate): {json.dumps(npvp, indent=2, default=float)}")

    print("\n\nComputing scale-free IF table (this reruns similarity lookups + bootstraps, may take a bit)...")
    scale_free = compute_scale_free_table(data)
    json.dump(scale_free, open(BASE / "scale_free_if_table.json", "w"), indent=2, default=float)
    print(f"Wrote {BASE / 'scale_free_if_table.json'}")
    for ds in data:
        msf = matched_utility_scale_free(data, scale_free, ds)
        print(f"[{ds}] matched-utility PUFFIN-vs-curve (scale-free ratio): {json.dumps(msf, indent=2, default=float)}")
        npvp_sf = no_privacy_vs_plain_scale_free(scale_free, ds)
        print(f"[{ds}] no_privacy vs plain (scale-free ratio, lower=more fair): {json.dumps(npvp_sf, indent=2, default=float)}")
        mmr = matched_mean_risk_comparison(data, scale_free, ds)
        print(f"[{ds}] matched-mean-risk comparison (utility & decile0 at same mean risk): {json.dumps(mmr, indent=2, default=float)}")


if __name__ == "__main__":
    main()
