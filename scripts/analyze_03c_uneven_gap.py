#!/usr/bin/env python
"""Checkpoint 3c: is naive-vs-adaptive measurement error UNEVEN across users?

Pure analysis over already-saved reports/figures/phase3b/*/per_user_3b.csv
-- no new training. For each (dataset, mechanism) with mechanism in
{plain (lambda4=0.0), fairness-active (lambda4=1.0)}:
  1. measurement gap = post_adaptive_top1 - post_top1 (naive), by decile,
     with bootstrap CIs.
  2. Kruskal-Wallis test: is the gap's distribution the same across the
     10 entropy deciles, or does it vary (non-uniform measurement error)?
  3. Spearman rank correlation between naive privacy_gain and adaptive
     privacy_gain (ranking users by each); report how many users move
     from top-quartile-protected (naive) to bottom-quartile-protected
     (adaptive), and vice versa.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.metrics.privacy import privacy_gain

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
MECHANISMS = [("plain", "0.0"), ("fairness_active", "1.0")]


def analyze_one(ds, mech_label, l4):
    path = pathlib.Path(f"reports/figures/phase3b/{ds}_lambda4_{l4}/per_user_3b.csv")
    df = pd.read_csv(path)
    gap = (df["post_adaptive_top1"] - df["post_top1"]).to_numpy()
    deciles = df["entropy_decile"].to_numpy()

    gap_ci = bootstrap_decile_ci(gap, deciles)

    groups = [gap[deciles == d] for d in sorted(set(deciles)) if (deciles == d).sum() >= 2]
    h_stat, p_value = kruskal(*groups)

    naive_gain = privacy_gain(df["raw_top1"].to_numpy(), df["post_top1"].to_numpy())
    adaptive_gain = privacy_gain(df["raw_top1"].to_numpy(), df["post_adaptive_top1"].to_numpy())
    rho, rho_p = spearmanr(naive_gain, adaptive_gain)

    n = len(df)
    q = n // 4
    naive_rank = pd.Series(naive_gain).rank(ascending=False)  # rank 1 = best protected
    adaptive_rank = pd.Series(adaptive_gain).rank(ascending=False)
    top_quartile_naive = set(np.where(naive_rank <= q)[0])
    bottom_quartile_adaptive = set(np.where(adaptive_rank > n - q)[0])
    movers_well_to_poor = top_quartile_naive & bottom_quartile_adaptive

    bottom_quartile_naive = set(np.where(naive_rank > n - q)[0])
    top_quartile_adaptive = set(np.where(adaptive_rank <= q)[0])
    movers_poor_to_well = bottom_quartile_naive & top_quartile_adaptive

    # sensitivity: exclude near-zero raw_risk users (the flagged instability)
    stable = df["raw_top1"].to_numpy() >= 0.02
    rho_stable, rho_stable_p = spearmanr(naive_gain[stable], adaptive_gain[stable])

    return dict(
        dataset=ds, mechanism=mech_label, lambda4=l4, n_users=n,
        gap_ci_by_decile=gap_ci,
        kruskal_h=float(h_stat), kruskal_p=float(p_value),
        spearman_rho=float(rho), spearman_p=float(rho_p),
        spearman_rho_excl_near_zero_raw=float(rho_stable), spearman_p_excl_near_zero_raw=float(rho_stable_p),
        n_excluded_near_zero_raw=int((~stable).sum()),
        n_movers_well_to_poor=len(movers_well_to_poor), n_movers_poor_to_well=len(movers_poor_to_well),
        quartile_size=q,
    )


def main():
    results = []
    for ds in DATASETS:
        for mech_label, l4 in MECHANISMS:
            r = analyze_one(ds, mech_label, l4)
            results.append(r)
            print(f"{ds:12s} {mech_label:16s} n={r['n_users']:4d} "
                  f"kruskal_H={r['kruskal_h']:.2f} p={r['kruskal_p']:.2e} "
                  f"spearman_rho={r['spearman_rho']:.3f} (p={r['spearman_p']:.2e}) "
                  f"movers_well->poor={r['n_movers_well_to_poor']} poor->well={r['n_movers_poor_to_well']}")

    out_dir = pathlib.Path("reports/figures/phase3c")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
