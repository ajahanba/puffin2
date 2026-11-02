import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from puffin.attacks.analysis import paired_bootstrap_test

MIN_WINDOWS = 10
BASE = pathlib.Path("reports/figures/retraining_to_convergence")


def load(name):
    d = BASE / name
    summ = json.load(open(d / "summary.json"))
    df = pd.read_csv(d / "per_user_04d.csv")
    df = df[df.n_windows >= MIN_WINDOWS]
    return summ, df


def per_seed_decile_mean(df, n_seeds, decile):
    sub = df[df.entropy_decile == decile]
    return np.array([sub[f"post_top1_encseed{s}"].mean() for s in range(n_seeds)])


plain_summ, plain_df = load("geolife_plain_lambda0.0_10seed")
plain_util = np.array(plain_summ["utility_per_encoder_seed"])
plain_d0 = per_seed_decile_mean(plain_df, 10, 0)
plain_d9 = per_seed_decile_mean(plain_df, 10, 9)

out = {}
for l4 in (0.3, 1.0, 3.0):
    summ, df = load(f"geolife_ed_soft_lambda{l4}_10seed")
    util = np.array(summ["utility_per_encoder_seed"])
    d0 = per_seed_decile_mean(df, 10, 0)
    d9 = per_seed_decile_mean(df, 10, 9)

    util_test = paired_bootstrap_test(util, plain_util)
    d0_test = paired_bootstrap_test(plain_d0, d0)  # positive = ed_soft lower risk (good)
    d9_test = paired_bootstrap_test(plain_d9, d9)  # positive = ed_soft lower risk at d9 too

    dominates = bool(util_test["ci_excludes_zero"] and util_test["mean_diff"] > 0
                      and d0_test["ci_excludes_zero"] and d0_test["mean_diff"] > 0)
    d9_significantly_harmed = bool(d9_test["ci_excludes_zero"] and d9_test["mean_diff"] < 0)

    out[f"lambda4_{l4}"] = dict(
        utility_ed_soft=float(util.mean()), utility_plain=float(plain_util.mean()),
        util_test=util_test, decile0_test=d0_test, decile9_test=d9_test,
        dominates_at_10_seeds=dominates, decile9_significantly_harmed=d9_significantly_harmed,
    )
    print(f"lambda4={l4}: util ed_soft={util.mean():.4f} plain={plain_util.mean():.4f} "
          f"util_diff={util_test['mean_diff']:+.4f}[{util_test['ci_lo']:+.4f},{util_test['ci_hi']:+.4f}] excl0={util_test['ci_excludes_zero']}")
    print(f"  decile0: plain-ed_soft={d0_test['mean_diff']:+.4f}[{d0_test['ci_lo']:+.4f},{d0_test['ci_hi']:+.4f}] excl0={d0_test['ci_excludes_zero']}")
    print(f"  decile9: plain-ed_soft={d9_test['mean_diff']:+.4f}[{d9_test['ci_lo']:+.4f},{d9_test['ci_hi']:+.4f}] excl0={d9_test['ci_excludes_zero']} harmed={d9_significantly_harmed}")
    print(f"  dominates_at_10_seeds={dominates}")

out_path = pathlib.Path("reports/figures/retraining_to_convergence_analysis")
out_path.mkdir(parents=True, exist_ok=True)


def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


json.dump(out, open(out_path / "geolife_10seed_decile0_decile9.json", "w"), indent=2, default=default)
print(f"\nWrote {out_path / 'geolife_10seed_decile0_decile9.json'}")
