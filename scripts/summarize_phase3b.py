import json

for ds in ["geolife", "foursquare", "gowalla_nyc"]:
    for l4 in ["0.0", "0.3", "1.0", "3.0"]:
        p = f"reports/figures/phase3b/{ds}_lambda4_{l4}/summary.json"
        try:
            d = json.load(open(p))
            tg = d["target_group"]
            naive = tg["mean_post_naive_top1"]
            spec = tg["mean_post_specialized_top1"]
            rew = tg["mean_post_reweighted_top1"]
            n = tg["n_users"]
            sm = tg["specialized_multiplier"]
            rm = tg["reweighted_multiplier"]
            print(f"{ds:12s} l4={l4:4s} n={n:3d} naive={naive:.3f} spec={spec:.3f} rew={rew:.3f} "
                  f"spec_mult={sm:.2f} rew_mult={rm:.2f}")
        except FileNotFoundError:
            print(f"{ds:12s} l4={l4:4s} MISSING")
