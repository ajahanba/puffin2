#!/usr/bin/env python
"""Paper Section 8 (Disparate Risk): corrected decile-reduction metrics
for GeoLife and Foursquare's plain Mo-PAE, matching the format already
used for the Gowalla table. GeoLife uses the best available data (40
epochs, 10 encoder seeds); Foursquare uses the only available plain-
Mo-PAE run (20 epochs, 3 encoder seeds) -- noted in the Implementation
appendix, not in the main prose.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/data/mashhadi/puffin2")  # for remote execution

from puffin.metrics.decile_reduction import decile_reduction_ci, residual_risk_disparity, residual_risk_difference

MIN_WINDOWS = 10


def load_and_filter(path):
    df = pd.read_csv(path)
    return df[df["n_windows"] >= MIN_WINDOWS].copy()


def compute(df):
    raw = df["raw_top1"].to_numpy()
    post = df["post_top1"].to_numpy()
    dec = df["entropy_decile"].to_numpy()
    red = decile_reduction_ci(raw, post, dec)
    ratio = residual_risk_disparity(post, dec)
    diff = residual_risk_difference(post, dec)
    return dict(reduction=red, ratio=ratio, difference=diff, n_users=len(df))


def main():
    out = {}

    geolife_df = load_and_filter(
        "/data/mashhadi/puffin2/reports/figures/phase4e_10seed/geolife_plain_lambda0.0_10seed/per_user_04d.csv")
    out["geolife"] = compute(geolife_df)
    out["geolife"]["epochs"] = 40
    out["geolife"]["n_encoder_seeds"] = 10

    foursquare_df = load_and_filter(
        "/data/mashhadi/puffin2/reports/figures/phase4d/foursquare_plain_lambda0.0/per_user_04d.csv")
    out["foursquare"] = compute(foursquare_df)
    out["foursquare"]["epochs"] = 20
    out["foursquare"]["n_encoder_seeds"] = 3

    gowalla_df = load_and_filter(
        "/data/mashhadi/puffin2/reports/figures/phase4d/gowalla_nyc_plain_lambda0.0/per_user_04d.csv")
    out["gowalla_nyc"] = compute(gowalla_df)
    out["gowalla_nyc"]["epochs"] = 20
    out["gowalla_nyc"]["n_encoder_seeds"] = 3

    out_path = pathlib.Path("/data/mashhadi/puffin2/reports/figures/paper_section8/geolife_foursquare_disparate_risk.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))

    json.dump(out, open(out_path, "w"), indent=2, default=default)
    for ds in out:
        r = out[ds]
        d0 = r["reduction"][0]["relative_reduction"]
        d9 = r["reduction"][9]["relative_reduction"]
        print(f"{ds}: n={r['n_users']} epochs={r['epochs']} seeds={r['n_encoder_seeds']} "
              f"d0_rel_red={d0['mean']:.3f}[{d0['ci_lo']:.3f},{d0['ci_hi']:.3f}] "
              f"d9_rel_red={d9['mean']:.3f}[{d9['ci_lo']:.3f},{d9['ci_hi']:.3f}] "
              f"ratio={r['ratio']['point_estimate']:.2f}[{r['ratio']['ci_lo']:.2f},{r['ratio']['ci_hi']:.2f}] "
              f"diff={r['difference']['point_estimate']:.3f}[{r['difference']['ci_lo']:.3f},{r['difference']['ci_hi']:.3f}]")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
