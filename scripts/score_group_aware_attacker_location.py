#!/usr/bin/env python
"""Scores location-PUFFIN (lambda4=0.3) vs Mo-PAE under the group-aware
attacker, same procedure as scripts/score_group_aware_attacker.py: fixed decile
membership, the same candidate/target definition (candidates = decile
members with >=1 test window; targets = candidates with >=10 test
windows AND present in BOTH of the ORIGINAL group_aware_attacker.csv's
standard runs -- geolife_plain_lp1.2_10seed and
geolife_ed_soft_lambda1.0_10seed -- reused unchanged so the target
population is identical to group_aware_attacker.csv's, not recomputed
against location-PUFFIN's own standard run), and the same dual paired
bootstrap (one shared user-resample per replicate, reused for both the
group-attacker delta and the standard-attacker delta; attacker-seed/
encoder-seed axes resampled separately per arm).
"""
import pathlib

import numpy as np
import pandas as pd

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
OUT_DIR = pathlib.Path("reports/figures/paper")
N_BOOT = 2000
RNG = np.random.default_rng(0)

MOPAE_GROUP_LABEL = "mopae_matched_lambda3_1.2"
LOCATION_GROUP_LABEL = "puffin_location_lambda4_0.3"
MOPAE_STD_RUN = "geolife_plain_lp1.2_10seed"
ENTROPY_PUFFIN_STD_RUN = "geolife_ed_soft_lambda1.0_10seed"  # target definition only, unchanged from group_aware_attacker.csv

LOCATION_STD_RUN_10SEED = OUT_DIR / "geolife_ed_soft_location_gradmatched_lambda0.3_10seed"
LOCATION_STD_RUN_3SEED = OUT_DIR / "geolife_ed_soft_location_gradmatched_lambda0.3_3seed"
if (LOCATION_STD_RUN_10SEED / "summary.json").exists():
    LOCATION_STD_RUN = LOCATION_STD_RUN_10SEED
    print(f"Using the pooled 10-seed location-PUFFIN run: {LOCATION_STD_RUN}")
else:
    LOCATION_STD_RUN = LOCATION_STD_RUN_3SEED
    print(f"10-seed pooled location-PUFFIN run not found -- using the existing 3-seed run instead: "
          f"{LOCATION_STD_RUN}")


def load_group_csv(mech_label, decile):
    d = OUT_DIR / f"geolife_{mech_label}_decile{decile}_group_attacker"
    df = pd.read_csv(d / "per_user_04d.csv").set_index("user_id")
    seed_cols = sorted([c for c in df.columns if c.startswith("top1_seed")],
                        key=lambda c: int(c.replace("top1_seed", "")))
    return df, seed_cols


def paired_bootstrap_dual(risk_puffin_group, risk_mopae_group,
                           risk_puffin_std, risk_mopae_std, rng, n_boot=N_BOOT):
    n_users = risk_puffin_group.shape[1]
    assert risk_mopae_group.shape[1] == n_users
    assert risk_puffin_std.shape[1] == n_users
    assert risk_mopae_std.shape[1] == n_users
    group_deltas, std_deltas = [], []
    for _ in range(n_boot):
        user_boot = rng.integers(0, n_users, n_users)
        g_seed_p = rng.integers(0, risk_puffin_group.shape[0], risk_puffin_group.shape[0])
        g_seed_m = rng.integers(0, risk_mopae_group.shape[0], risk_mopae_group.shape[0])
        s_seed_p = rng.integers(0, risk_puffin_std.shape[0], risk_puffin_std.shape[0])
        s_seed_m = rng.integers(0, risk_mopae_std.shape[0], risk_mopae_std.shape[0])

        gp = risk_puffin_group[g_seed_p][:, user_boot].mean(axis=0)
        gm = risk_mopae_group[g_seed_m][:, user_boot].mean(axis=0)
        group_deltas.append((gp - gm).mean())

        sp = risk_puffin_std[s_seed_p][:, user_boot].mean(axis=0)
        sm = risk_mopae_std[s_seed_m][:, user_boot].mean(axis=0)
        std_deltas.append((sp - sm).mean())

    group_deltas = np.array(group_deltas)
    std_deltas = np.array(std_deltas)
    g_lo, g_hi = np.percentile(group_deltas, [2.5, 97.5])
    s_lo, s_hi = np.percentile(std_deltas, [2.5, 97.5])
    return (dict(mean_delta=float(group_deltas.mean()), ci_lo=float(g_lo), ci_hi=float(g_hi)),
            dict(mean_delta=float(std_deltas.mean()), ci_lo=float(s_lo), ci_hi=float(s_hi)))


# Target definition reuses the ORIGINAL group_aware_attacker.csv standard runs, unchanged.
std_mopae_full = pd.read_csv(RETRAIN_BASE / MOPAE_STD_RUN / "per_user_04d.csv").set_index("user_id")
std_entropy_puffin_full = pd.read_csv(RETRAIN_BASE / ENTROPY_PUFFIN_STD_RUN / "per_user_04d.csv").set_index("user_id")

std_seed_cols_m = sorted([c for c in std_mopae_full.columns if c.startswith("post_top1_encseed")],
                          key=lambda c: int(c.replace("post_top1_encseed", "")))

location_std_full = pd.read_csv(LOCATION_STD_RUN / "per_user_04d.csv").set_index("user_id")
loc_seed_cols = sorted([c for c in location_std_full.columns if c.startswith("post_top1_encseed")],
                        key=lambda c: int(c.replace("post_top1_encseed", "")))

rows = []
for d in range(10):
    pm, seed_cols_m = load_group_csv(MOPAE_GROUP_LABEL, d)
    pp, seed_cols_p = load_group_csv(LOCATION_GROUP_LABEL, d)
    candidates = sorted(set(pm.index) & set(pp.index))
    assert set(pm.index) == set(pp.index), (
        f"decile {d}: group-attacker candidate sets differ between mechanisms "
        f"({set(pm.index) ^ set(pp.index)})")
    n_candidates = len(candidates)
    chance = 1.0 / n_candidates if n_candidates else None

    # Targets: same definition as group_aware_attacker.csv (candidates with
    # >=10 test windows AND present in BOTH of the ORIGINAL std runs).
    enough_windows = {u for u in candidates if pm.loc[u, "n_windows"] >= 10}
    dropped_low_window = sorted(set(candidates) - enough_windows)
    in_both_std = enough_windows & set(std_mopae_full.index) & set(std_entropy_puffin_full.index)
    dropped_std_missing = sorted(enough_windows - in_both_std)
    targets = sorted(in_both_std)

    # Location's own standard run may be missing a handful of these targets
    # (same kind of population mismatch group_aware_attacker.csv's docstring
    # describes for the two original std runs); drop those few, noted below.
    dropped_location_missing = sorted(set(targets) - set(location_std_full.index))
    targets = sorted(set(targets) - set(dropped_location_missing))
    n_targets = len(targets)

    risk_puffin_group = pp.loc[targets, seed_cols_p].to_numpy().T
    risk_mopae_group = pm.loc[targets, seed_cols_m].to_numpy().T
    risk_puffin_std = location_std_full.loc[targets, loc_seed_cols].to_numpy().T
    risk_mopae_std = std_mopae_full.loc[targets, std_seed_cols_m].to_numpy().T

    group_res, std_res = paired_bootstrap_dual(
        risk_puffin_group, risk_mopae_group, risk_puffin_std, risk_mopae_std, RNG, n_boot=N_BOOT)

    rows.append(dict(
        decile=int(d), n_candidates=n_candidates, n_targets=n_targets, chance_level=chance,
        dropped_low_window=",".join(str(u) for u in dropped_low_window) or None,
        dropped_std_missing=",".join(str(u) for u in dropped_std_missing) or None,
        dropped_location_missing=",".join(str(u) for u in dropped_location_missing) or None,
        puffin_group_seed_min=float(risk_puffin_group.mean(axis=1).min()),
        puffin_group_seed_max=float(risk_puffin_group.mean(axis=1).max()),
        mopae_group_seed_min=float(risk_mopae_group.mean(axis=1).min()),
        mopae_group_seed_max=float(risk_mopae_group.mean(axis=1).max()),
        puffin_std_seed_min=float(risk_puffin_std.mean(axis=1).min()),
        puffin_std_seed_max=float(risk_puffin_std.mean(axis=1).max()),
        mopae_std_seed_min=float(risk_mopae_std.mean(axis=1).min()),
        mopae_std_seed_max=float(risk_mopae_std.mean(axis=1).max()),
        group_attacker_delta=group_res["mean_delta"], group_attacker_ci_lo=group_res["ci_lo"],
        group_attacker_ci_hi=group_res["ci_hi"],
        standard_attacker_delta=std_res["mean_delta"], standard_attacker_ci_lo=std_res["ci_lo"],
        standard_attacker_ci_hi=std_res["ci_hi"],
    ))
    print(f"decile {d}: candidates={n_candidates}, targets={n_targets}, "
          f"dropped(low_window)={dropped_low_window}, dropped(std_missing)={dropped_std_missing}, "
          f"dropped(location_missing)={dropped_location_missing}, chance={chance:.4f}, "
          f"group_delta={group_res['mean_delta']:+.4f}[{group_res['ci_lo']:+.4f},{group_res['ci_hi']:+.4f}], "
          f"standard_delta={std_res['mean_delta']:+.4f}[{std_res['ci_lo']:+.4f},{std_res['ci_hi']:+.4f}]",
          flush=True)

out_df = pd.DataFrame(rows)
out_path = OUT_DIR / "group_aware_attacker_location.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}, {len(out_df)} rows (std run used: {LOCATION_STD_RUN})")
