#!/usr/bin/env python
"""GeoLife ed_soft vs. plain Mo-PAE, post min_epochs-fix, two analyses
per the PI's explicit instruction (2026-11-02), replacing the retracted
"dominance" framing entirely:

1. Direct paired comparison (same lambda_priv=0.3, ed_soft's 10 encoder
   seeds vs. plain's 10 encoder seeds): decile-0 risk, decile-9 risk,
   utility, and mean risk over all users, each as a paired-bootstrap
   effect size with a 95% CI. This is Table 7's main content.

2. Frontier test: is each ed_soft point's decile-0/9/mean/max-min risk
   lower than what the plain-Mo-PAE lambda3 curve achieves AT THE SAME
   (interpolated) utility -- i.e., is ed_soft doing something a simple
   privacy-weight turn-up on plain Mo-PAE cannot reach? Bootstrapped
   over both the ed_soft seeds and the curve's own per-point seeds
   jointly, per bootstrap draw (not just a point-estimate interpolation).
   Bootstrap draws where ed_soft's resampled utility falls outside the
   resampled curve's own range are dropped (extrapolation refused) and
   counted, not guessed at.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from puffin.attacks.analysis import paired_bootstrap_test

MIN_WINDOWS = 10
LAMBDA3_SWEEP = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8]
BASE = pathlib.Path("reports/figures/retraining_to_convergence")
N_BOOT = 5000
RNG = np.random.default_rng(0)


def load(name):
    d = BASE / name
    summ = json.load(open(d / "summary.json"))
    df = pd.read_csv(d / "per_user_04d.csv")
    df = df[df.n_windows >= MIN_WINDOWS].copy()
    return summ, df


def per_seed_metrics(df, n_seeds):
    decile0 = np.array([df.loc[df.entropy_decile == 0, f"post_top1_encseed{s}"].mean() for s in range(n_seeds)])
    decile9 = np.array([df.loc[df.entropy_decile == 9, f"post_top1_encseed{s}"].mean() for s in range(n_seeds)])
    mean_all = np.array([df[f"post_top1_encseed{s}"].mean() for s in range(n_seeds)])
    maxmin = np.array([
        df.groupby("entropy_decile")[f"post_top1_encseed{s}"].mean().pipe(lambda x: x.max() - x.min())
        for s in range(n_seeds)
    ])
    return dict(decile0=decile0, decile9=decile9, mean_all=mean_all, maxmin=maxmin)


def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


# ---- load curve points (plain, lambda_priv swept) ----
curve_points = []
for lp in LAMBDA3_SWEEP:
    name = "geolife_plain_lambda0.0_10seed" if lp == 0.3 else f"geolife_plain_lp{lp}"
    summ, df = load(name)
    n_seeds = len(summ["encoder_seeds"])
    util_per_seed = np.array(summ["utility_per_encoder_seed"])
    metrics = per_seed_metrics(df, n_seeds)
    curve_points.append(dict(lambda3=lp, util=util_per_seed, n_seeds=n_seeds, source=name, **metrics))

# ---- load ed_soft points ----
ed_soft_points = {}
plain_10seed_summ, plain_10seed_df = load("geolife_plain_lambda0.0_10seed")
plain_metrics = per_seed_metrics(plain_10seed_df, len(plain_10seed_summ["encoder_seeds"]))
plain_util = np.array(plain_10seed_summ["utility_per_encoder_seed"])

for l4 in (0.3, 1.0, 3.0):
    summ, df = load(f"geolife_ed_soft_lambda{l4}_10seed")
    n_seeds = len(summ["encoder_seeds"])
    util_per_seed = np.array(summ["utility_per_encoder_seed"])
    metrics = per_seed_metrics(df, n_seeds)
    ed_soft_points[l4] = dict(util=util_per_seed, n_seeds=n_seeds, **metrics)

# ---- analysis 1: direct paired comparison (Table 7 main content) ----
direct = {}
for l4, pt in ed_soft_points.items():
    util_test = paired_bootstrap_test(pt["util"], plain_util)
    d0_test = paired_bootstrap_test(plain_metrics["decile0"], pt["decile0"])  # + = ed_soft lower risk
    d9_test = paired_bootstrap_test(plain_metrics["decile9"], pt["decile9"])  # + = ed_soft lower risk
    mean_test = paired_bootstrap_test(plain_metrics["mean_all"], pt["mean_all"])  # + = ed_soft lower risk
    direct[f"lambda4_{l4}"] = dict(utility_test=util_test, decile0_risk_test=d0_test,
                                     decile9_risk_test=d9_test, mean_risk_test=mean_test)
    print(f"[direct] lambda4={l4}:")
    print(f"  utility (ed_soft-plain): {util_test['mean_diff']:+.4f} [{util_test['ci_lo']:+.4f},{util_test['ci_hi']:+.4f}] excl0={util_test['ci_excludes_zero']}")
    print(f"  decile0 risk (plain-ed_soft): {d0_test['mean_diff']:+.4f} [{d0_test['ci_lo']:+.4f},{d0_test['ci_hi']:+.4f}] excl0={d0_test['ci_excludes_zero']}")
    print(f"  decile9 risk (plain-ed_soft): {d9_test['mean_diff']:+.4f} [{d9_test['ci_lo']:+.4f},{d9_test['ci_hi']:+.4f}] excl0={d9_test['ci_excludes_zero']}")
    print(f"  mean risk, all users (plain-ed_soft): {mean_test['mean_diff']:+.4f} [{mean_test['ci_lo']:+.4f},{mean_test['ci_hi']:+.4f}] excl0={mean_test['ci_excludes_zero']}")

# ---- analysis 2: frontier test (curve-interpolated, jointly bootstrapped) ----
def bootstrap_frontier(metric_key):
    out = {}
    for l4, pt in ed_soft_points.items():
        deltas = []
        n_extrap = 0
        for _ in range(N_BOOT):
            idx_t = RNG.integers(0, pt["n_seeds"], pt["n_seeds"])
            util_t = pt["util"][idx_t].mean()
            metric_t = pt[metric_key][idx_t].mean()
            curve_utils_b, curve_metrics_b = [], []
            for cp in curve_points:
                idx_c = RNG.integers(0, cp["n_seeds"], cp["n_seeds"])
                curve_utils_b.append(cp["util"][idx_c].mean())
                curve_metrics_b.append(cp[metric_key][idx_c].mean())
            order = np.argsort(curve_utils_b)
            cu = np.array(curve_utils_b)[order]
            cm = np.array(curve_metrics_b)[order]
            if util_t < cu[0] or util_t > cu[-1]:
                n_extrap += 1
                continue
            interp_metric = np.interp(util_t, cu, cm)
            deltas.append(interp_metric - metric_t)  # + = ed_soft below curve (lower risk than curve can reach)
        deltas = np.array(deltas)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5]) if len(deltas) else (np.nan, np.nan)
        out[l4] = dict(mean_delta=float(deltas.mean()) if len(deltas) else None,
                        ci_lo=float(ci_lo) if len(deltas) else None, ci_hi=float(ci_hi) if len(deltas) else None,
                        ci_excludes_zero=bool(len(deltas) and (ci_lo > 0 or ci_hi < 0)),
                        n_valid_boot=int(len(deltas)), n_extrapolated=int(n_extrap))
    return out

frontier = {}
for metric_key in ("decile0", "decile9", "mean_all", "maxmin"):
    frontier[metric_key] = bootstrap_frontier(metric_key)
    print(f"\n[frontier:{metric_key}]")
    for l4, r in frontier[metric_key].items():
        if r["mean_delta"] is None:
            print(f"  lambda4={l4}: ALL {N_BOOT} draws required extrapolation -- unresolvable")
        else:
            print(f"  lambda4={l4}: delta={r['mean_delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
                  f"excl0={r['ci_excludes_zero']} (extrapolated {r['n_extrapolated']}/{N_BOOT})")

# ---- also report: ed_soft decile0 risk vs the curve's global minimum decile0 risk
# (the strongest form of "something plain can't reach at any weight") ----
print("\n[global minimum check: decile0]")
curve_decile0_means = [cp["decile0"].mean() for cp in curve_points]
curve_min_idx = int(np.argmin(curve_decile0_means))
print(f"  curve's lowest decile0 risk: {curve_decile0_means[curve_min_idx]:.4f} at lambda3={curve_points[curve_min_idx]['lambda3']} (utility {curve_points[curve_min_idx]['util'].mean():.4f})")
for l4, pt in ed_soft_points.items():
    print(f"  ed_soft lambda4={l4}: decile0 risk {pt['decile0'].mean():.4f} (utility {pt['util'].mean():.4f}) "
          f"-- {'BELOW' if pt['decile0'].mean() < curve_decile0_means[curve_min_idx] else 'at or above'} curve's global minimum")

out_path = pathlib.Path("reports/figures/retraining_to_convergence_analysis")
out_path.mkdir(parents=True, exist_ok=True)
json.dump(dict(direct=direct, frontier=frontier,
                curve_points=[dict(lambda3=cp["lambda3"], utility_mean=float(cp["util"].mean()),
                                    decile0_mean=float(cp["decile0"].mean()), decile9_mean=float(cp["decile9"].mean()),
                                    mean_all_mean=float(cp["mean_all"].mean()), maxmin_mean=float(cp["maxmin"].mean()),
                                    n_seeds=cp["n_seeds"], source=cp["source"]) for cp in curve_points]),
          open(out_path / "geolife_frontier_test.json", "w"), indent=2, default=default)
print(f"\nWrote {out_path / 'geolife_frontier_test.json'}")
