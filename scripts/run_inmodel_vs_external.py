#!/usr/bin/env python
"""Per dataset, for Mo-PAE
lambda3=0.3 and PUFFIN lambda4=1.0, mean in-model URU confidence in the
true user vs. the external attacker's top-1, overall and for deciles 0
and 9, plus chance level (1/#users). One table.
"""
import json
import pathlib

import pandas as pd

MIN_WINDOWS = 10
RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")

RUNS = {
    "geolife": {
        "Mo-PAE (lambda3=0.3)": "geolife_plain_lambda0.0_10seed",
        "PUFFIN (lambda4=1.0)": "geolife_ed_soft_lambda1.0_10seed",
    },
    "foursquare": {
        "Mo-PAE (lambda3=0.3)": "foursquare_plain_lp0.3",
        "PUFFIN (lambda4=1.0)": "foursquare_ed_soft_lambda1.0",
    },
    "gowalla_sf_bay": {
        "Mo-PAE (lambda3=0.3)": "gowalla_sf_bay_plain_lp0.3",
        "PUFFIN (lambda4=1.0)": "gowalla_sf_bay_ed_soft_lambda1.0",
    },
}

rows = []
for ds, mechs in RUNS.items():
    for label, run_name in mechs.items():
        run_dir = RETRAIN_BASE / run_name
        summ = json.load(open(run_dir / "summary.json"))
        df = pd.read_csv(run_dir / "per_user_04d.csv")
        df = df[df.n_windows >= MIN_WINDOWS].copy()
        total_users = summ.get("n_users_before_filter", len(df))
        chance = 1.0 / total_users

        for scope, sub in (("overall", df), ("decile0", df[df.entropy_decile == 0]),
                            ("decile9", df[df.entropy_decile == 9])):
            rows.append(dict(
                dataset=ds, mechanism=label, source=run_name, scope=scope, n_users=len(sub),
                inmodel_confidence_mean=float(sub["inmodel_confidence"].mean()) if len(sub) else None,
                external_top1_mean=float(sub["post_top1"].mean()) if len(sub) else None,
                chance_level=chance,
            ))
        print(f"[{ds}] {label} ({run_name}): total_users={total_users}, chance={chance:.5f}", flush=True)

out_df = pd.DataFrame(rows)
out_path = "reports/figures/paper/inmodel_vs_external.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}, {len(out_df)} rows")
print(out_df.to_string())
