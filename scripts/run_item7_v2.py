#!/usr/bin/env python
"""Item 7 REDO (PI correction, 2026-11-09): the higher absolute risk
under the group-restricted attacker is mostly the smaller candidate set
(about 18 users vs. 178, so chance is ~10x higher) -- NOT evidence for
the candidate-pool finding retracted in the adaptive-attacker appendix.
What matters is PUFFIN vs. Mo-PAE under the SAME group attacker. Per
decile: chance level, per-seed group-attacker risk for both arms, and
a paired user bootstrap CI (corrected pairing, scripts/paired_bootstrap_v2.py)
for PUFFIN - Mo-PAE under the group attacker AND under the standard
(full-population) attacker. Saves per-user CSVs in the standard format.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_attacker_on_frozen_encoder_strongest_per_user
from puffin.fairness.losses import build_entropy_table
from scripts.run_phase4d_approved_batch import train_mechanism, eval_utility
from scripts.paired_bootstrap_v2 import paired_bootstrap_matched

DEVICE = torch.device("cuda:1")
PARQUET = "data/processed/geolife/unified.parquet"
RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
OUT_DIR = pathlib.Path("reports/figures/paper")
ATTACKER_SEEDS = (0, 1, 2, 3, 4)
ENCODER_SEED = 0
N_BOOT = 2000
RNG = np.random.default_rng(0)

MECHS = [
    dict(label="mopae_matched_lambda3_1.2", mechanism="plain", lambda4=0.0, lambda_priv=1.2,
         standard_run="geolife_plain_lp1.2_10seed"),
    dict(label="puffin_lambda4_1.0", mechanism="ed_soft", lambda4=1.0, lambda_priv=0.3,
         standard_run="geolife_ed_soft_lambda1.0_10seed"),
]


def filter_windows_to_users(windows, user_idx_0based_set):
    out = {}
    for split in ("train", "test"):
        w = windows[split]
        mask = np.isin(w["user"], list(user_idx_0based_set))
        out[split] = {k: v[mask] for k, v in w.items()}
    return out


def direct_paired_bootstrap(risk_a, risk_b, rng, n_boot=N_BOOT):
    """risk_a, risk_b: (n_seeds, n_users), SAME user alignment. delta = A - B,
    one shared user draw per replicate, seeds resampled independently
    per arm (each arm's own attacker-seed axis)."""
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


def main():
    df = pd.read_parquet(PARQUET)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=DEVICE)

    print("building windows...", flush=True)
    windows = build_sl_windows(df, sl=10, max_windows_per_user=200, respect_traj_boundary=False)

    feats_decile = feats.copy()
    feats_decile["entropy_decile"] = pd.qcut(feats_decile["s_unc"], 10, labels=False, duplicates="drop")
    decile_by_user_1idx = dict(zip(feats_decile["user_id"], feats_decile["entropy_decile"]))
    decile_by_user_0idx = {u - 1: d for u, d in decile_by_user_1idx.items()}

    group_post = {}  # [label][decile] -> post_df (user_id, post_top1_encseedN...)
    bundles = {}
    for mech in MECHS:
        print(f"\n=== {mech['label']}: training encoder seed={ENCODER_SEED} ===", flush=True)
        t0 = time.time()
        bundle, actual_epochs = train_mechanism(
            windows, total_locations, total_users, entropy_table, mech["mechanism"], mech["lambda4"],
            DEVICE, max_epochs=80, lambda_priv=mech["lambda_priv"], seed=ENCODER_SEED,
            patience=8, val_frac=0.1, min_epochs=20)
        utility = eval_utility(bundle, windows, DEVICE)
        print(f"{mech['label']}: stopped after {actual_epochs} epochs, utility={utility:.4f}, "
              f"trained in {time.time()-t0:.1f}s", flush=True)
        bundles[mech["label"]] = bundle
        group_post[mech["label"]] = {}

        for d in sorted(set(decile_by_user_0idx.values())):
            decile_users_0idx = {u for u, dd in decile_by_user_0idx.items() if dd == d}
            dec_windows = filter_windows_to_users(windows, decile_users_0idx)
            if len(dec_windows["test"]["user"]) == 0:
                continue
            post_df, _ = train_attacker_on_frozen_encoder_strongest_per_user(
                bundle.encoder, dec_windows, total_users, DEVICE, seeds=ATTACKER_SEEDS,
                max_epochs=25, patience=6, attacker_device=torch.device("cpu"))
            group_post[mech["label"]][d] = post_df
            out_name = f"geolife_{mech['label']}_decile{d}_group_attacker"
            out_dir = OUT_DIR / out_name
            out_dir.mkdir(parents=True, exist_ok=True)
            post_df.to_csv(out_dir / "per_user_04d.csv", index=False)
            print(f"  decile {d}: saved {out_dir}/per_user_04d.csv, n_users={len(decile_users_0idx)}", flush=True)

    # --- build the comparison table ---
    rows = []
    mopae_label, puffin_label = MECHS[0]["label"], MECHS[1]["label"]
    for d in sorted(set(decile_by_user_0idx.values())):
        if d not in group_post[mopae_label] or d not in group_post[puffin_label]:
            continue
        pm = group_post[mopae_label][d].set_index("user_id")
        pp = group_post[puffin_label][d].set_index("user_id")
        common = sorted(set(pm.index) & set(pp.index))
        n_group = len(common)
        chance = 1.0 / n_group if n_group else None

        seed_cols = [c for c in pm.columns if c.startswith("post_top1_encseed")]
        risk_mopae_group = pm.loc[common, seed_cols].to_numpy().T
        risk_puffin_group = pp.loc[common, seed_cols].to_numpy().T

        group_res = direct_paired_bootstrap(risk_puffin_group, risk_mopae_group, RNG, n_boot=N_BOOT)

        # standard (full-population) attacker, restricted to this decile's users
        std_mopae = pd.read_csv(RETRAIN_BASE / MECHS[0]["standard_run"] / "per_user_04d.csv").set_index("user_id")
        std_puffin = pd.read_csv(RETRAIN_BASE / MECHS[1]["standard_run"] / "per_user_04d.csv").set_index("user_id")
        std_seed_cols_m = sorted([c for c in std_mopae.columns if c.startswith("post_top1_encseed")],
                                  key=lambda c: int(c.replace("post_top1_encseed", "")))
        std_seed_cols_p = sorted([c for c in std_puffin.columns if c.startswith("post_top1_encseed")],
                                  key=lambda c: int(c.replace("post_top1_encseed", "")))
        # group_post's user_id is ALREADY 1-indexed (puffin.attacks.reid._per_user_topk
        # does user_id=int(u)+1 internally), same convention as the standard per_user_04d.csv --
        # confirmed, no further offset needed.
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
        print(f"decile {d}: chance={chance:.4f}, group_delta={group_res['mean_delta']:+.4f}"
              f"[{group_res['ci_lo']:+.4f},{group_res['ci_hi']:+.4f}], "
              f"standard_delta={std_res['mean_delta']}", flush=True)

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "item7_group_aware_attacker_v2.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}, {len(out_df)} rows")


if __name__ == "__main__":
    main()
