#!/usr/bin/env python
"""Checkpoint 3d: decomposes the naive-vs-adaptive measurement gap for one
dataset's fairness-active Mo-PAE mechanism (lambda4=1.0, the same object
Checkpoint 3/3b's headline 2.4-2.8x claim was about).

Trains the mechanism once (seed=0), then computes (a)-(e) (see
puffin.attacks.gap_decomposition's module docstring) across 5 attacker
seeds each, averaging per user -- same division of labor as every other
checkpoint (one trained mechanism, multi-seed attacker). Applies the
per-user minimum (>=10 test windows) before any per-user metric or
bootstrap, and reports how many users were excluded.
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
from puffin.attacks.adaptive import (
    user_decile_map, group_reweighted_attack_multiseed, per_group_specialized_attack_multiseed,
)
from puffin.attacks.gap_decomposition import (
    train_naive_with_logits, naive_per_user, naive_restricted_per_user,
    train_per_group_heads, specialized_per_user_oracle,
    train_decile_classifier, specialized_per_user_realistic,
)
from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.fairness.losses import ed_soft, build_entropy_table
from puffin.models.mopae_torch import build_bundle, fixed_train_step


def train_mopae_fair(windows, total_locations, total_users, entropy_table, lambda_priv, lambda4, device,
                      epochs, batch_size=512, lr=3e-4, seed=0):
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


def _run_seed_a_b(encoder, windows, total_users, decile_map, device, attack_epochs, seed):
    logits, te_y, te_user_1idx = train_naive_with_logits(encoder, windows, total_users, device,
                                                           epochs=attack_epochs, seed=seed)
    a = naive_per_user(logits, te_y, te_user_1idx).rename(columns={"correct": f"a_seed{seed}"})
    b = naive_restricted_per_user(logits, te_y, te_user_1idx, decile_map).rename(columns={"correct": f"b_seed{seed}"})
    return a, b


def _run_seed_c_e(encoder, windows, feats, total_users, device, attack_epochs, seed):
    heads, decile_map = train_per_group_heads(encoder, windows, feats, total_users, device,
                                                epochs=attack_epochs, seed=seed)
    c = specialized_per_user_oracle(encoder, windows, heads, decile_map, device).rename(
        columns={"correct": f"c_seed{seed}"})
    classifier, _ = train_decile_classifier(encoder, windows, feats, device, epochs=attack_epochs, seed=seed)
    e = specialized_per_user_realistic(encoder, windows, heads, classifier, device).rename(
        columns={"correct": f"e_seed{seed}"})
    return c, e


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
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--reweight-factor", type=float, default=10.0)
    ap.add_argument("--min-test-windows", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    tag = f"{args.dataset} gap-decomp"

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)
    decile_map = user_decile_map(feats, 10)

    print(f"[{tag}] building windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)

    print(f"[{tag}] training mechanism (seed=0, lambda4={args.lambda4})...", flush=True)
    bundle = train_mopae_fair(windows, total_locations, total_users, entropy_table, args.lambda_priv,
                               args.lambda4, device, epochs=args.mopae_epochs)
    encoder = bundle.encoder

    a_dfs, b_dfs, c_dfs, e_dfs = [], [], [], []
    for s in seeds:
        print(f"[{tag}] seed {s}: (a)/(b)...", flush=True)
        a, b = _run_seed_a_b(encoder, windows, total_users, decile_map, device, args.attack_epochs, s)
        a_dfs.append(a)
        b_dfs.append(b)
        print(f"[{tag}] seed {s}: (c)/(e)...", flush=True)
        c, e = _run_seed_c_e(encoder, windows, feats, total_users, device, args.attack_epochs, s)
        c_dfs.append(c)
        e_dfs.append(e)

    print(f"[{tag}] (d) reweighted attack (needs a target decile -- using the lowest-entropy "
          f"decile, pre-specified, per the PI's item-4 instruction applied here too)...", flush=True)
    d_df = group_reweighted_attack_multiseed(encoder, windows, feats, total_users, device,
                                              target_deciles=[0], seeds=seeds, epochs=args.attack_epochs)
    d_df = d_df.rename(columns={"reweighted_top1": "d"})[["user_id", "d"]]

    def merge_seeds(dfs, letter):
        out = dfs[0]
        for other in dfs[1:]:
            out = out.merge(other, on="user_id", how="outer")
        cols = [c for c in out.columns if c.startswith(f"{letter}_seed")]
        out[letter] = out[cols].mean(axis=1)
        return out[["user_id", letter]]

    a_df = merge_seeds(a_dfs, "a")
    b_df = merge_seeds(b_dfs, "b")
    c_df = merge_seeds(c_dfs, "c")
    e_df = merge_seeds(e_dfs, "e")

    merged = feats[["user_id", "s_unc"]].copy()
    merged["entropy_decile"] = merged["user_id"].map(decile_map)
    n_windows = pd.DataFrame({"user_id": windows["test"]["user"] + 1}).value_counts().reset_index(name="n_test_windows")
    merged = merged.merge(n_windows, on="user_id", how="left")
    merged["n_test_windows"] = merged["n_test_windows"].fillna(0).astype(int)
    for df_, col in [(a_df, "a"), (b_df, "b"), (c_df, "c"), (e_df, "e"), (d_df, "d")]:
        merged = merged.merge(df_, on="user_id", how="left")

    n_before = len(merged)
    excluded = merged[merged["n_test_windows"] < args.min_test_windows]
    merged = merged[merged["n_test_windows"] >= args.min_test_windows].dropna(subset=["a", "b", "c", "e", "d"])
    n_excluded = n_before - len(merged)
    print(f"[{tag}] per-user minimum filter: excluded {n_excluded}/{n_before} users "
          f"(<{args.min_test_windows} test windows)", flush=True)

    deciles = merged["entropy_decile"].to_numpy()
    components = {}
    for name, vals in [("a_naive", merged["a"]), ("b_restricted", merged["b"]),
                        ("c_specialized", merged["c"]), ("d_reweighted", merged["d"]),
                        ("e_realistic", merged["e"]),
                        ("max_c_d", merged[["c", "d"]].max(axis=1))]:
        components[name] = bootstrap_decile_ci(vals.to_numpy(), deciles)

    gaps = {}
    for name, diff in [("side_information_b_minus_a", merged["b"] - merged["a"]),
                        ("specialization_c_minus_b", merged["c"] - merged["b"]),
                        ("realistic_e_minus_a", merged["e"] - merged["a"])]:
        gaps[name] = bootstrap_decile_ci(diff.to_numpy(), deciles)

    result = dict(
        dataset=args.dataset, lambda4=args.lambda4, n_users_before_filter=n_before,
        n_users_excluded_min_windows=n_excluded, min_test_windows=args.min_test_windows,
        components_by_decile=components, gap_components_by_decile=gaps,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "per_user_3d.csv", index=False)
    excluded.to_csv(out_dir / "excluded_users.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("components_by_decile", "gap_components_by_decile")}, indent=2))


if __name__ == "__main__":
    main()
