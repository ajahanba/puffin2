#!/usr/bin/env python
"""Phase 3 task 5 (last): does a fairness-aware adaptive adversary recover
MORE risk for the group a mechanism protects most than a group-agnostic
attacker does?

1. Train a fairness-ACTIVE Mo-PAE mechanism (ed_soft, lambda4>0 -- unlike
   Checkpoints 2/2b/2c's plain-Mo-PAE/lambda4=0 runs, this one is actually
   trying to be fair).
2. Find which entropy decile(s) it protects most, using the ordinary
   group-agnostic attacker (puffin.attacks.reid) -- the "most protected"
   group by puffin.metrics.privacy.privacy_gain.
3. Attack the SAME frozen encoder with both adaptive variants
   (puffin.attacks.adaptive) and compare, FOR THAT SAME GROUP, against the
   group-agnostic attacker's risk: does knowing the group recover more?
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
from puffin.attacks.reid import train_raw_attacker_hardened, train_attacker_on_frozen_encoder_hardened
from puffin.attacks.adaptive import user_decile_map, per_group_specialized_attack, group_reweighted_attack
from puffin.metrics.privacy import privacy_gain
from puffin.models.mopae_torch import build_bundle, fixed_train_step
from puffin.fairness.losses import ed_soft, build_entropy_table


def train_mopae_fair(windows, total_locations, total_users, entropy_table, lambda_priv, lambda4, device,
                      epochs, batch_size=512, lr=3e-4, seed=0):
    """Same shape as run_phase2_disparate_risk.py's train_mopae_plain, but
    with the fairness term ACTIVE (fairness_fn=ed_soft, lambda4>0) -- that
    script deliberately keeps lambda4=0 throughout (the PI's "plain Mo-PAE
    first" instruction); this one is the fairness-mechanism-under-test for
    the adaptive adversary, so it needs the fairness term switched on."""
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                              coord_target[idx], fairness_fn=ed_soft, entropy_table=entropy_table,
                              lambda4=lambda4, lambdas=lambdas)
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--mopae-epochs", type=int, default=20)
    ap.add_argument("--lambda-priv", type=float, default=0.3)
    ap.add_argument("--lambda4", type=float, default=1.0)
    ap.add_argument("--attack-epochs", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--reweight-factor", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = tuple(int(s) for s in args.seeds.split(","))

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

    print(f"[{args.dataset}] building SL={args.sl} windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)

    print(f"[{args.dataset}] training FAIR Mo-PAE (lambda4={args.lambda4}, "
          f"lambda_priv={args.lambda_priv})...", flush=True)
    bundle = train_mopae_fair(windows, total_locations, total_users, entropy_table,
                               args.lambda_priv, args.lambda4, device, epochs=args.mopae_epochs)

    print(f"[{args.dataset}] global (group-agnostic) raw + post attacker...", flush=True)
    raw_df, _ = train_raw_attacker_hardened(windows, total_locations, total_users, device, seeds=seeds,
                                             max_epochs=args.attack_epochs, patience=6)
    post_df = train_attacker_on_frozen_encoder_hardened(bundle.encoder, windows, total_users, device,
                                                         seeds=seeds, max_epochs=args.attack_epochs, patience=6)

    merged = feats.merge(raw_df, on="user_id", how="inner").merge(post_df, on="user_id", how="inner")
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")
    merged["global_gain"] = privacy_gain(merged["raw_top1"].to_numpy(), merged["post_top1"].to_numpy())
    decile_gain = merged.groupby("entropy_decile")["global_gain"].mean().sort_values(ascending=False)
    most_protected = [int(decile_gain.index[0])]
    print(f"[{args.dataset}] decile mean global_gain (sorted): {decile_gain.to_dict()}", flush=True)
    print(f"[{args.dataset}] most-protected decile(s): {most_protected}", flush=True)

    print(f"[{args.dataset}] per-group specialized adversary...", flush=True)
    spec_df = per_group_specialized_attack(bundle.encoder, windows, feats, total_users, device,
                                            epochs=args.attack_epochs)
    print(f"[{args.dataset}] group-reweighted adversary (target={most_protected})...", flush=True)
    rew_df = group_reweighted_attack(bundle.encoder, windows, feats, total_users, device,
                                      target_deciles=most_protected, reweight_factor=args.reweight_factor,
                                      epochs=args.attack_epochs)

    combined = merged.merge(spec_df[["user_id", "specialized_top1"]], on="user_id", how="left")
    combined = combined.merge(rew_df[["user_id", "reweighted_top1"]], on="user_id", how="left")
    combined["specialized_vs_global_delta"] = combined["specialized_top1"] - combined["post_top1"]
    combined["reweighted_vs_global_delta"] = combined["reweighted_top1"] - combined["post_top1"]

    target_rows = combined[combined["entropy_decile"].isin(most_protected)]
    summary = dict(
        dataset=args.dataset, n_users=len(combined), most_protected_decile=most_protected,
        decile_mean_global_gain={int(k): float(v) for k, v in decile_gain.to_dict().items()},
        target_group=dict(
            n_users=len(target_rows),
            mean_global_post_top1=float(target_rows["post_top1"].mean()),
            mean_specialized_top1=float(target_rows["specialized_top1"].mean()),
            mean_reweighted_top1=float(target_rows["reweighted_top1"].mean()),
            mean_specialized_delta=float(target_rows["specialized_vs_global_delta"].mean()),
            mean_reweighted_delta=float(target_rows["reweighted_vs_global_delta"].mean()),
        ),
        all_users=dict(
            mean_specialized_delta=float(combined["specialized_vs_global_delta"].mean()),
            mean_reweighted_delta=float(combined["reweighted_vs_global_delta"].mean()),
        ),
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "per_user_adaptive.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
