import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from puffin.attacks.analysis import bootstrap_decile_ci

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
FORMULATIONS = ["ed_soft", "lipschitz", "floor", "variance", "group_dro", "per_user_minimax", "group_adversarial"]

results = []
for ds in DATASETS:
    for f in FORMULATIONS:
        p = pathlib.Path(f"reports/figures/phase4/{ds}_{f}/per_user_phase4.csv")
        summ_p = pathlib.Path(f"reports/figures/phase4/{ds}_{f}/summary.json")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        summ = json.load(open(summ_p))
        worst_decile = summ["worst_off_decile_by_adaptive_attacker"]
        gap = (df["post_adaptive_top1"] - df["post_top1"]).to_numpy()
        deciles = df["entropy_decile"].to_numpy()
        ci = bootstrap_decile_ci(gap, deciles)
        g = ci.get(worst_decile, ci.get(str(worst_decile)))
        results.append(dict(dataset=ds, formulation=f, worst_decile=worst_decile,
                             utility=summ["utility_next_location_top1"], gap=g))
        print(f"{ds:12s} {f:18s} utility={summ['utility_next_location_top1']:.3f} "
              f"worst_decile={worst_decile} gap={g['mean']:+.3f} CI=[{g['ci_lo']:+.3f},{g['ci_hi']:+.3f}] "
              f"excl0={g['ci_excludes_zero']}")

pathlib.Path("reports/figures/phase4").mkdir(parents=True, exist_ok=True)
json.dump(results, open("reports/figures/phase4/gap_ci_summary.json", "w"), indent=2)
