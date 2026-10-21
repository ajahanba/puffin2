import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from puffin.metrics.decile_reduction import decile_reduction_ci, residual_risk_disparity

DATASETS = ["gowalla_nyc", "gowalla_la", "gowalla_chicago", "gowalla_sf_bay"]

results = {}
for ds in DATASETS:
    df = pd.read_csv(f"reports/figures/phase2d/{ds}/per_user_filtered.csv")
    raw = df["raw_top1"].to_numpy()
    post = df["post_top1"].to_numpy()
    deciles = df["entropy_decile"].to_numpy()
    red = decile_reduction_ci(raw, post, deciles)
    disp = residual_risk_disparity(post, deciles, low_decile=0, high_decile=9)
    results[ds] = dict(decile_reduction=red, residual_risk_disparity=disp)
    print(f"=== {ds} ===")
    for d in sorted(red.keys()):
        r = red[d]
        print(f"  decile {d}: raw={r['raw_mean']:.3f} post={r['post_mean']:.3f} "
              f"abs_red={r['absolute_reduction']['mean']:+.3f} "
              f"[{r['absolute_reduction']['ci_lo']:+.3f},{r['absolute_reduction']['ci_hi']:+.3f}] "
              f"rel_red={r['relative_reduction']['mean']:.3f} "
              f"[{r['relative_reduction']['ci_lo']:.3f},{r['relative_reduction']['ci_hi']:.3f}]")
    print(f"  residual_risk_disparity (decile0/decile9): {disp['point_estimate']:.2f}x "
          f"[{disp['ci_lo']:.2f},{disp['ci_hi']:.2f}]")

pathlib.Path("reports/figures/phase2d/decile_reduction_metrics.json").write_text(json.dumps(results, indent=2))
