#!/usr/bin/env python
"""Recomputes item 7's comparison table from the already-saved per-user
CSVs (training succeeded; the comparison-building loop crashed on a
column-name mismatch -- group-attacker CSVs use top1_seed{N}, the
ATTACKER seed axis, not post_top1_encseed{N})."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
OUT_DIR = pathlib.Path("reports/figures/paper")
N_BOOT = 2000
RNG = np.random.default_rng(0)

MOPAE_STD_RUN = "geolife_plain_lp1.2_10seed"
PUFFIN_STD_RUN = "geolife_ed_soft_lambda1.0_10seed"


def direct_paired_bootstrap(risk_a, risk_b, rng, n_boot=N_BOOT):
    n_users = risk_a.shape[1]
    deltas = []
    for _ in range(n_boot):
        user_boot = rng.integers(0, n_users, n_users)
        seed_boot_a = rng.integers(0, risk_a.shape[0], risk_a.shape[0])
        seed_boot_b = rng.integers(0, risk_b.shape[0], risk_b.shape[0])
        a_b = risk_a[seed_boot_a][:, user_boot].mean(axis=0)
        b_b = risk_b[seed_boot_b][:, user_boot].mean(axis=0)
        deltas.append((a_b - b_b).mean())
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return dict(mean_delta=float(deltas.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi))


def load_group_csv(mech_label, decile):
    d = OUT_DIR / f"geolife_{mech_label}_decile{decile}_group_attacker"
    df = pd.read_csv(d / "per_user_04d.csv").set_index("user_id")
    seed_cols = sorted([c for c in df.columns if c.startswith("top1_seed")],
                        key=lambda c: int(c.replace("top1_seed", "")))
    return df, seed_cols


rows = []
for d in range(10):
    pm, seed_cols_m = load_group_csv("mopae_matched_lambda3_1.2", d)
    pp, seed_cols_p = load_group_csv("puffin_lambda4_1.0", d)
    common = sorted(set(pm.index) & set(pp.index))
    n_group = len(common)
    chance = 1.0 / n_group if n_group else None

    risk_mopae_group = pm.loc[common, seed_cols_m].to_numpy().T
    risk_puffin_group = pp.loc[common, seed_cols_p].to_numpy().T

    group_res = direct_paired_bootstrap(risk_puffin_group, risk_mopae_group, RNG, n_boot=N_BOOT)

    std_mopae = pd.read_csv(RETRAIN_BASE / MOPAE_STD_RUN / "per_user_04d.csv").set_index("user_id")
    std_puffin = pd.read_csv(RETRAIN_BASE / PUFFIN_STD_RUN / "per_user_04d.csv").set_index("user_id")
    std_seed_cols_m = sorted([c for c in std_mopae.columns if c.startswith("post_top1_encseed")],
                              key=lambda c: int(c.replace("post_top1_encseed", "")))
    std_seed_cols_p = sorted([c for c in std_puffin.columns if c.startswith("post_top1_encseed")],
                              key=lambda c: int(c.replace("post_top1_encseed", "")))
    std_common = sorted(set(std_mopae.index) & set(std_puffin.index) & set(common))
    if len(std_common) >= 5:
        risk_mopae_std = std_mopae.loc[std_common, std_seed_cols_m].to_numpy().T
        risk_puffin_std = std_puffin.loc[std_common, std_seed_cols_p].to_numpy().T
        std_res = direct_paired_bootstrap(risk_puffin_std, risk_mopae_std, RNG, n_boot=N_BOOT)
    else:
        std_res = dict(mean_delta=None, ci_lo=None, ci_hi=None)

    rows.append(dict(
        decile=int(d), n_users_in_group=n_group, chance_level=chance,
        puffin_group_seed_min=float(risk_puffin_group.mean(axis=1).min()),
        puffin_group_seed_max=float(risk_puffin_group.mean(axis=1).max()),
        mopae_group_seed_min=float(risk_mopae_group.mean(axis=1).min()),
        mopae_group_seed_max=float(risk_mopae_group.mean(axis=1).max()),
        group_attacker_delta=group_res["mean_delta"], group_attacker_ci_lo=group_res["ci_lo"],
        group_attacker_ci_hi=group_res["ci_hi"],
        standard_attacker_delta=std_res["mean_delta"], standard_attacker_ci_lo=std_res["ci_lo"],
        standard_attacker_ci_hi=std_res["ci_hi"], n_users_standard=len(std_common),
    ))
    print(f"decile {d}: n_group={n_group}, chance={chance:.4f}, "
          f"group_delta={group_res['mean_delta']:+.4f}[{group_res['ci_lo']:+.4f},{group_res['ci_hi']:+.4f}], "
          f"standard_delta={std_res['mean_delta']:+.4f}[{std_res['ci_lo']:+.4f},{std_res['ci_hi']:+.4f}]"
          if std_res["mean_delta"] is not None else
          f"decile {d}: n_group={n_group}, chance={chance:.4f}, "
          f"group_delta={group_res['mean_delta']:+.4f}[{group_res['ci_lo']:+.4f},{group_res['ci_hi']:+.4f}], "
          f"standard_delta=N/A", flush=True)

out_df = pd.DataFrame(rows)
out_path = OUT_DIR / "item7_group_aware_attacker_v2.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}, {len(out_df)} rows")
