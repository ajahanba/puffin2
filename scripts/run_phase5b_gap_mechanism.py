#!/usr/bin/env python
"""Checkpoint 5b (positive control, item 2 of the PI's latest instruction):
is the uneven naive-vs-adaptive measurement gap specific to LEARNED
encoders (Mo-PAE), or does it also appear for mechanisms that have no
learned encoder at all -- GI-DP (a fixed, unlearned perturbation) and
TrajGAN (a learned GENERATOR, but not a privacy-preserving encoder in
Mo-PAE's sense)?

One (dataset, mechanism) combination per invocation. mechanism in
{gi_dp, trajgan} -- "plain" Mo-PAE needs no new run here, it's already
fully covered by Checkpoint 3b's lambda4=0.0 data (reused directly in the
05b report, not rerun).

For gi_dp: perturbs every row with Planar Laplace noise at the given
epsilon, builds windows from the perturbed release, computes the naive
attacker's accuracy directly on it (puffin.attacks.reid's own hardened
multi-seed attacker -- "post/naive" exactly as for Mo-PAE, just attacking
perturbed trajectories instead of a learned encoder's features), trains
a single attacker-side encoder on the perturbed release
(puffin.attacks.mechanism_adapter), then runs the SAME multi-seed
specialized/reweighted adaptive attacks on top of it Checkpoint 3b used.

For trajgan: trains the GAN on the real train split, generates one
synthetic counterpart of every real test trajectory (that's ALL TrajGAN
releases -- no synthetic train data exists), then further splits those
synthetic trajectories 80/20 by user into an attacker-train/attacker-test
set (a second, nested split, needed because the attacker needs something
to learn from that isn't the same data it's evaluated on) before running
the same naive + adaptive pipeline on that nested split.

Raw risk (pre-mechanism) is NOT recomputed here -- reused directly from
Checkpoint 3b's per_user_3b.csv for the same dataset (same raw attacker,
same data, already computed under full 5-seed rigor).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_hardened
from puffin.attacks.adaptive import per_group_specialized_attack_multiseed, group_reweighted_attack_multiseed
from puffin.attacks.mechanism_adapter import train_encoder_on_released_windows
from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.metrics.privacy import privacy_gain
from puffin.baselines.gi_dp import run_gi_dp


def _finish_eval(windows, total_locations, total_users, device, feats, raw_df, args, tag):
    print(f"[{tag}] naive attacker on released data...", flush=True)
    naive_df, _ = train_raw_attacker_hardened(windows, total_locations, total_users, device,
                                               seeds=tuple(int(s) for s in args.global_seeds.split(",")),
                                               max_epochs=args.attack_epochs, patience=6)
    naive_df = naive_df.rename(columns={"raw_top1": "post_top1", "raw_top1_std": "post_top1_std"})

    merged = feats.merge(raw_df[["user_id", "raw_top1"]], on="user_id", how="inner")
    merged = merged.merge(naive_df[["user_id", "post_top1"]], on="user_id", how="inner")
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")
    merged["global_gain"] = privacy_gain(merged["raw_top1"].to_numpy(), merged["post_top1"].to_numpy())
    decile_gain = merged.groupby("entropy_decile")["global_gain"].mean().sort_values(ascending=False)
    most_protected = [int(decile_gain.index[0])]

    print(f"[{tag}] training attacker-side encoder on released data...", flush=True)
    encoder = train_encoder_on_released_windows(windows, total_locations, total_users, device,
                                                 epochs=args.attack_epochs, seed=0)

    print(f"[{tag}] adaptive attacks, {args.adaptive_seeds} seeds each...", flush=True)
    adaptive_seeds = tuple(int(s) for s in args.adaptive_seeds.split(","))
    spec_df = per_group_specialized_attack_multiseed(encoder, windows, feats, total_users, device,
                                                       seeds=adaptive_seeds, epochs=args.attack_epochs)
    rew_df = group_reweighted_attack_multiseed(encoder, windows, feats, total_users, device,
                                                target_deciles=most_protected, reweight_factor=args.reweight_factor,
                                                seeds=adaptive_seeds, epochs=args.attack_epochs)
    combined = merged.merge(spec_df[["user_id", "specialized_top1"]], on="user_id", how="left")
    combined = combined.merge(rew_df[["user_id", "reweighted_top1"]], on="user_id", how="left")
    combined["post_adaptive_top1"] = combined[["specialized_top1", "reweighted_top1"]].max(axis=1)
    deciles = combined["entropy_decile"].to_numpy()

    gap = (combined["post_adaptive_top1"] - combined["post_top1"]).to_numpy()
    gap_ci = bootstrap_decile_ci(gap, deciles)
    result = dict(
        mechanism=args.mechanism, dataset=args.dataset, n_users=len(combined),
        most_protected_decile=most_protected,
        gap_ci_by_decile=gap_ci,
        decile_mean_naive=merged.groupby("entropy_decile")["post_top1"].mean().to_dict(),
    )
    return combined, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mechanism", required=True, choices=["gi_dp", "trajgan"])
    ap.add_argument("--epsilon", type=float, default=0.01, help="gi_dp only")
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--global-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--adaptive-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--reweight-factor", type=float, default=10.0)
    ap.add_argument("--raw-attacker-csv", required=True,
                     help="path to an existing per_user_3b.csv (Checkpoint 3b) for this dataset, "
                          "reused for raw_top1 instead of recomputing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    tag = f"{args.dataset} {args.mechanism}" + (f" eps={args.epsilon}" if args.mechanism == "gi_dp" else "")

    df = pd.read_parquet(args.parquet)
    feats = compute_mobility_features(df)
    raw_df = pd.read_csv(args.raw_attacker_csv)[["user_id", "raw_top1"]]

    if args.mechanism == "gi_dp":
        print(f"[{tag}] perturbing with Planar Laplace, epsilon={args.epsilon}...", flush=True)
        perturbed = run_gi_dp(df, epsilon=args.epsilon, seed=0)
        perturbed = perturbed.merge(df[["user_id", "traj_id", "ts", "split"]],
                                     on=["user_id", "traj_id", "ts"], how="inner")
        total_locations = int(perturbed["cell_id"].max())
        total_users = int(df["user_id"].max())
        windows = build_sl_windows(perturbed, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                    respect_traj_boundary=args.respect_traj_boundary)
    else:  # trajgan
        from puffin.baselines.trajgan import run_trajgan
        print(f"[{tag}] training TrajGAN and generating synthetic test trajectories...", flush=True)
        synthetic_df, train_summary = run_trajgan(df, device=device, max_epochs=30, seed=0)
        print(f"[{tag}] trajgan train summary: n_train={train_summary['n_train_trajectories']} "
              f"n_test={train_summary['n_test_trajectories']}", flush=True)
        # nested 80/20 split of the SYNTHETIC trajectories by user, since that's all
        # TrajGAN releases -- the attacker needs its own train/test within it.
        rng = np.random.default_rng(0)
        users = synthetic_df["user_id"].unique()
        rng.shuffle(users)
        n_train_users = int(len(users) * 0.8)
        train_users = set(users[:n_train_users])
        synthetic_df["split"] = synthetic_df["user_id"].apply(lambda u: "train" if u in train_users else "test")
        total_locations = int(synthetic_df["cell_id"].max())
        total_users = int(df["user_id"].max())
        windows = build_sl_windows(synthetic_df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                    respect_traj_boundary=args.respect_traj_boundary)

    combined, result = _finish_eval(windows, total_locations, total_users, device, feats, raw_df, args, tag)
    result["epsilon"] = args.epsilon if args.mechanism == "gi_dp" else None

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "per_user_5b.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "gap_ci_by_decile"}, indent=2))


if __name__ == "__main__":
    main()
