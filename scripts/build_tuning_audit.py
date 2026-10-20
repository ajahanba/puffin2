import json
import pathlib

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
FORMULATIONS = ["ed_soft", "lipschitz", "floor", "variance", "group_dro", "per_user_minimax", "group_adversarial"]

rows = []
# ed_soft's real tuning history lives in phase3b (lambda4 in {0,0.3,1.0,3.0}, 5 seeds, matches
# the original PUFFIN L_pair_threshold baseline under its real name there)
for ds in DATASETS:
    for l4 in ["0.0", "0.3", "1.0", "3.0"]:
        p = pathlib.Path(f"reports/figures/phase3b/{ds}_lambda4_{l4}/summary.json")
        if p.exists():
            d = json.load(open(p))
            rows.append(dict(formulation="ed_soft", dataset=ds, lambda4=float(l4), seeds=5,
                              source=f"phase3b/{ds}_lambda4_{l4}"))

# phase4 grid + follow-ups for everything else (and ed_soft's own phase4 rerun, redundant with
# phase3b's lambda4=1.0 but kept for completeness)
for ds in DATASETS:
    for f in FORMULATIONS:
        for suffix in ["", "_lambda0.3"]:
            p = pathlib.Path(f"reports/figures/phase4/{ds}_{f}{suffix}/summary.json")
            if p.exists():
                d = json.load(open(p))
                rows.append(dict(formulation=f, dataset=ds, lambda4=d["lambda4"], seeds=5,
                                  source=f"phase4/{ds}_{f}{suffix}"))

# dedupe identical (formulation, dataset, lambda4) entries, keep first
seen = set()
deduped = []
for r in rows:
    key = (r["formulation"], r["dataset"], r["lambda4"])
    if key in seen:
        continue
    seen.add(key)
    deduped.append(r)

print(f"{'formulation':18s} {'dataset':12s} {'lambda4':8s} {'seeds':6s} source")
for r in sorted(deduped, key=lambda r: (r["formulation"], r["dataset"], r["lambda4"])):
    print(f"{r['formulation']:18s} {r['dataset']:12s} {r['lambda4']:8.1f} {r['seeds']:6d} {r['source']}")

print()
print("=== lambda points per formulation (pooled across datasets) ===")
by_formulation = {}
for r in deduped:
    by_formulation.setdefault(r["formulation"], set()).add(r["lambda4"])
for f, points in sorted(by_formulation.items()):
    print(f"{f:18s} n_points={len(points):2d} points={sorted(points)}")

json.dump(deduped, open("reports/figures/phase4/tuning_audit.json", "w"), indent=2)
