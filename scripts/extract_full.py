import json

print("=== delta CIs (specialized/reweighted - naive, target decile) ===")
for ds in ["geolife", "foursquare", "gowalla_nyc"]:
    for l4 in ["0.0", "0.3", "1.0", "3.0"]:
        p = f"reports/figures/phase3b/{ds}_lambda4_{l4}/summary.json"
        try:
            d = json.load(open(p))
            tg = d["target_group"]
            sc = tg["specialized_delta_ci"]
            rc = tg["reweighted_delta_ci"]
            sk = list(sc.keys())[0]
            rk = list(rc.keys())[0]
            s = sc[sk]
            r = rc[rk]
            print(f"{ds:12s} l4={l4:4s} spec_delta_ci=[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}] "
                  f"excl0={s['ci_excludes_zero']}  rew_delta_ci=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] excl0={r['ci_excludes_zero']}")
        except FileNotFoundError:
            print(f"{ds:12s} l4={l4:4s} MISSING")

print()
print("=== threat-model check + MIA (lambda4=1.0 runs only) ===")
for ds in ["geolife", "foursquare", "gowalla_nyc"]:
    p = f"reports/figures/phase3b/{ds}_lambda4_1.0/summary.json"
    d = json.load(open(p))
    tm = d.get("threat_model_decile_inference")
    mia = d.get("mia_auc")
    print(f"{ds}: threat_model={tm}")
    print(f"{ds}: mia={mia}")

print()
print("=== 3-stage decile table (lambda4=1.0 runs, all deciles) ===")
for ds in ["geolife", "foursquare", "gowalla_nyc"]:
    p = f"reports/figures/phase3b/{ds}_lambda4_1.0/summary.json"
    d = json.load(open(p))
    ts = d["three_stage_decile_risk_ci"]
    print(f"--- {ds} ---")
    deciles = sorted(ts["raw"].keys(), key=lambda x: int(x))
    for dec in deciles:
        raw = ts["raw"][dec]["mean"]
        naive = ts["post_naive"][dec]["mean"]
        adap = ts["post_adaptive"][dec]["mean"]
        print(f"  decile {dec}: raw={raw:.3f} post_naive={naive:.3f} post_adaptive={adap:.3f}")
