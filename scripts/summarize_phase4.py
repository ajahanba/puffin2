import json

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
FORMULATIONS = ["ed_soft", "lipschitz", "floor", "variance", "group_dro", "per_user_minimax", "group_adversarial"]

print(f"{'dataset':12s} {'formulation':18s} {'utility':8s} {'worst_decile':12s} {'naive':8s} {'adaptive':8s} {'gap':8s}")
rows = []
for ds in DATASETS:
    for f in FORMULATIONS:
        p = f"reports/figures/phase4/{ds}_{f}/summary.json"
        try:
            d = json.load(open(p))
            util = d["utility_next_location_top1"]
            wd = d["worst_off_decile_by_adaptive_attacker"]
            naive = d["worst_off_decile_naive_risk"]["mean"]
            adaptive = d["worst_off_decile_adaptive_risk"]["mean"]
            adaptive_ci = (d["worst_off_decile_adaptive_risk"]["ci_lo"], d["worst_off_decile_adaptive_risk"]["ci_hi"])
            rows.append(dict(dataset=ds, formulation=f, utility=util, worst_decile=wd,
                              naive=naive, adaptive=adaptive, adaptive_ci=adaptive_ci, gap=adaptive - naive))
            print(f"{ds:12s} {f:18s} {util:8.4f} {wd:12d} {naive:8.3f} {adaptive:8.3f} {adaptive-naive:+8.3f}")
        except FileNotFoundError:
            print(f"{ds:12s} {f:18s} MISSING")

json.dump(rows, open("reports/figures/phase4/combined_summary.json", "w"), indent=2)
