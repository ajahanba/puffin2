#!/usr/bin/env python
"""Item 7 v3 (PI instructions, 2026-11-10) -- a RE-SCORING pass over v2's
already-saved per-user CSVs, no retraining.

Two inconsistencies in v2's comparison table:
(a) v2 scored every CANDIDATE in a decile's group attacker (anyone with
    >=1 test window), including users with as few as 7 test windows
    (decile 0: users 73, 179) -- below the project's standard >=10-test-
    window reporting floor used everywhere else.
(b) v2's "standard attacker" column read directly from the two pre-
    existing standalone 10-seed runs (geolife_plain_lp1.2_10seed,
    geolife_ed_soft_lambda1.0_10seed). Those two runs were launched
    separately, 3 days apart (Oct 31 / Nov 2), under different git
    commits -- their own summary.json shows n_users_before_filter=180
    vs 181 (a mismatch that predates any windowing in this script: the
    underlying unified.parquet has been unchanged since Oct 16, and
    puffin/attacks/windowing.py did not change between those two
    commits either, so the discrepancy traces to a difference in how
    the two standalone runs were individually launched -- e.g. a
    respect-traj-boundary or sl/max-windows-per-user flag passed
    differently between the two separate invocations; neither run's
    logs recorded its exact CLI args, so the specific flag can't be
    recovered after the fact). The practical effect: user 88 (21 test
    windows, a candidate in BOTH mechanisms' freshly-trained decile-0
    group attacker -- confirmed identical candidate sets there) has a
    row in the plain/MoPAE standard run but NO row at all in the
    ed_soft/PUFFIN standard run. Checked the full population of both
    standard runs end-to-end: user 88 is the ONLY discrepancy (179 vs
    178 users, symmetric difference = {88} exactly) -- no other user is
    dropped by either standard run relative to its counterpart.

Fix: decile membership is unchanged from v2 (computed once, globally,
from training-split entropy over ALL GeoLife users -- puffin.features.
mobility.compute_mobility_features(df) on the full unified parquet,
same qcut call shared by both mechanisms' group-attacker training, so
it was already applied uniformly, not recomputed per run). Candidates
per decile: unchanged (all users with >=1 test window in that decile,
per the freshly-trained group attacker -- already identical between
mechanisms, verified). Targets per decile: candidates with >=10 test
windows (n_windows column, already test-split-only -- see puffin.
attacks.reid._per_user_topk) AND present in BOTH mechanisms' standard
10-seed runs. Only targets are scored, under both the group-restricted
and the standard (full-population) attacker. The paired bootstrap now
draws ONE shared user-resample (over the fixed target list) and reuses
it for BOTH the group-attacker delta and the standard-attacker delta
at a given replicate -- "same draw for both arms" per the PI's
instruction -- with attacker-seed axes resampled separately per arm
since the group attacker (5 attacker seeds) and the standard run (10
encoder seeds) do not share a seed space.
"""
import pathlib

import numpy as np
import pandas as pd

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
OUT_DIR = pathlib.Path("reports/figures/paper")
N_BOOT = 2000
RNG = np.random.default_rng(0)

MOPAE_GROUP_LABEL = "mopae_matched_lambda3_1.2"
PUFFIN_GROUP_LABEL = "puffin_lambda4_1.0"
MOPAE_STD_RUN = "geolife_plain_lp1.2_10seed"
PUFFIN_STD_RUN = "geolife_ed_soft_lambda1.0_10seed"


def load_group_csv(mech_label, decile):
    d = OUT_DIR / f"geolife_{mech_label}_decile{decile}_group_attacker"
    df = pd.read_csv(d / "per_user_04d.csv").set_index("user_id")
    seed_cols = sorted([c for c in df.columns if c.startswith("top1_seed")],
                        key=lambda c: int(c.replace("top1_seed", "")))
    return df, seed_cols


def paired_bootstrap_dual(risk_puffin_group, risk_mopae_group,
                           risk_puffin_std, risk_mopae_std, rng, n_boot=N_BOOT):
    """All four risk matrices are (n_seeds, n_targets), SAME target/user
    alignment across all four (group and std share the same target
    list by construction here). One shared user-resample draw per
    replicate, reused for both the group delta and the std delta;
    attacker-seed axes resampled independently per arm (different seed
    spaces: 5 attacker seeds for group, 10 encoder seeds for std)."""
    n_users = risk_puffin_group.shape[1]
    assert risk_mopae_group.shape[1] == n_users
    assert risk_puffin_std.shape[1] == n_users
    assert risk_mopae_std.shape[1] == n_users
    group_deltas, std_deltas = [], []
    for _ in range(n_boot):
        user_boot = rng.integers(0, n_users, n_users)  # SAME draw for both arms
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


# --- sanity check: confirm the ONLY discrepancy between the two standard
# runs' full populations is user 88 (per the explanation above) ---
std_mopae_full = pd.read_csv(RETRAIN_BASE / MOPAE_STD_RUN / "per_user_04d.csv").set_index("user_id")
std_puffin_full = pd.read_csv(RETRAIN_BASE / PUFFIN_STD_RUN / "per_user_04d.csv").set_index("user_id")
sym_diff = set(std_mopae_full.index) ^ set(std_puffin_full.index)
print(f"Standard-run population check: mopae n={len(std_mopae_full)}, puffin n={len(std_puffin_full)}, "
      f"symmetric difference={sorted(sym_diff)}")
assert sym_diff == {88}, f"expected only user 88 to differ, got {sym_diff}"

std_seed_cols_m = sorted([c for c in std_mopae_full.columns if c.startswith("post_top1_encseed")],
                          key=lambda c: int(c.replace("post_top1_encseed", "")))
std_seed_cols_p = sorted([c for c in std_puffin_full.columns if c.startswith("post_top1_encseed")],
                          key=lambda c: int(c.replace("post_top1_encseed", "")))

rows = []
for d in range(10):
    pm, seed_cols_m = load_group_csv(MOPAE_GROUP_LABEL, d)
    pp, seed_cols_p = load_group_csv(PUFFIN_GROUP_LABEL, d)
    candidates = sorted(set(pm.index) & set(pp.index))
    assert set(pm.index) == set(pp.index), (
        f"decile {d}: group-attacker candidate sets differ between mechanisms "
        f"({set(pm.index) ^ set(pp.index)}) -- was consistent in v2, should still be")
    n_candidates = len(candidates)
    chance = 1.0 / n_candidates if n_candidates else None

    # targets: candidates with >=10 test windows AND present in BOTH standard runs
    enough_windows = {u for u in candidates if pm.loc[u, "n_windows"] >= 10}
    dropped_low_window = sorted(set(candidates) - enough_windows)
    in_both_std = enough_windows & set(std_mopae_full.index) & set(std_puffin_full.index)
    dropped_std_missing = sorted(enough_windows - in_both_std)
    targets = sorted(in_both_std)
    n_targets = len(targets)

    risk_puffin_group = pp.loc[targets, seed_cols_p].to_numpy().T
    risk_mopae_group = pm.loc[targets, seed_cols_m].to_numpy().T
    risk_puffin_std = std_puffin_full.loc[targets, std_seed_cols_p].to_numpy().T
    risk_mopae_std = std_mopae_full.loc[targets, std_seed_cols_m].to_numpy().T

    group_res, std_res = paired_bootstrap_dual(
        risk_puffin_group, risk_mopae_group, risk_puffin_std, risk_mopae_std, RNG, n_boot=N_BOOT)

    rows.append(dict(
        decile=int(d), n_candidates=n_candidates, n_targets=n_targets, chance_level=chance,
        dropped_low_window=",".join(str(u) for u in dropped_low_window) or None,
        dropped_std_missing=",".join(str(u) for u in dropped_std_missing) or None,
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
          f"chance={chance:.4f}, "
          f"group_delta={group_res['mean_delta']:+.4f}[{group_res['ci_lo']:+.4f},{group_res['ci_hi']:+.4f}], "
          f"standard_delta={std_res['mean_delta']:+.4f}[{std_res['ci_lo']:+.4f},{std_res['ci_hi']:+.4f}]",
          flush=True)

out_df = pd.DataFrame(rows)
out_path = OUT_DIR / "item7_group_aware_attacker_v3.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}, {len(out_df)} rows")
