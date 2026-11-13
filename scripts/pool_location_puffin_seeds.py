#!/usr/bin/env python
"""Pools the existing 3-seed location-PUFFIN (gradient-matched) run with
the new 7-seed (seeds 3-9) run into a single 10-seed directory, matching
the standard '_10seed' output shape used elsewhere. Does not touch either
input run; writes a new output directory only.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from puffin.attacks.analysis import bootstrap_decile_ci

BASE = pathlib.Path("reports/figures/paper")

ap = argparse.ArgumentParser()
ap.add_argument("--lambda4", required=True)
args = ap.parse_args()
l4 = args.lambda4

dir_3seed = BASE / f"geolife_ed_soft_location_gradmatched_lambda{l4}_3seed"
dir_7seed = BASE / f"geolife_ed_soft_location_gradmatched_lambda{l4}_seeds39"
out_dir = BASE / f"geolife_ed_soft_location_gradmatched_lambda{l4}_10seed"
out_dir.mkdir(parents=True, exist_ok=True)

df3 = pd.read_csv(dir_3seed / "per_user_04d.csv")
df7 = pd.read_csv(dir_7seed / "per_user_04d.csv")
s3 = json.loads((dir_3seed / "summary.json").read_text())
s7 = json.loads((dir_7seed / "summary.json").read_text())

assert s3["encoder_seeds"] == [0, 1, 2], s3["encoder_seeds"]
assert s7["encoder_seeds"] == [3, 4, 5, 6, 7, 8, 9], s7["encoder_seeds"]

shared_cols = ["user_id", "n_windows", "raw_top1", "s_unc"]
shared_cols = [c for c in shared_cols if c in df3.columns and c in df7.columns]
merged = df3[shared_cols].merge(df7[shared_cols], on="user_id", suffixes=("_3seed", "_7seed"))
for c in shared_cols:
    if c == "user_id":
        continue
    a, b = merged[f"{c}_3seed"], merged[f"{c}_7seed"]
    if not np.allclose(a, b, equal_nan=True):
        raise AssertionError(f"{c} differs between the 3-seed and 7-seed runs for some users")

df3_cols = ["user_id"] + [c for c in df3.columns if c != "user_id" and c not in (
    "post_top1", "post_top1_std_over_encoder_seeds", "inmodel_confidence")]
df7_cols = ["user_id"] + [c for c in df7.columns if c != "user_id" and (
    c.startswith("post_top1_encseed") or c.startswith("inmodel_confidence_encseed"))]
df = df3[df3_cols].merge(df7[df7_cols], on="user_id", how="inner")

post_cols = sorted([c for c in df.columns if c.startswith("post_top1_encseed")],
                    key=lambda c: int(c.replace("post_top1_encseed", "")))
df["post_top1"] = df[post_cols].mean(axis=1)
df["post_top1_std_over_encoder_seeds"] = df[post_cols].std(axis=1)
inmodel_cols = sorted([c for c in df.columns if c.startswith("inmodel_confidence_encseed")],
                       key=lambda c: int(c.replace("inmodel_confidence_encseed", "")))
if inmodel_cols:
    df["inmodel_confidence"] = df[inmodel_cols].mean(axis=1)

df.to_csv(out_dir / "per_user_04d.csv", index=False)

deciles = df["entropy_decile"].to_numpy()
raw_ci = bootstrap_decile_ci(df["raw_top1"].to_numpy(), deciles)
post_ci = bootstrap_decile_ci(df["post_top1"].to_numpy(), deciles)
inmodel_ci = bootstrap_decile_ci(df["inmodel_confidence"].to_numpy(), deciles) if inmodel_cols else None

encoder_seeds = s3["encoder_seeds"] + s7["encoder_seeds"]
utility_per_encoder_seed = s3["utility_per_encoder_seed"] + s7["utility_per_encoder_seed"]
actual_epochs_per_encoder_seed = s3["actual_epochs_per_encoder_seed"] + s7["actual_epochs_per_encoder_seed"]
train_wall_time = s3.get("train_wall_time_seconds_per_encoder_seed", []) + s7.get(
    "train_wall_time_seconds_per_encoder_seed", [])

summary = dict(
    dataset=s3["dataset"], mechanism=s3["mechanism"], lambda4=s3["lambda4"], lambda_priv=s3["lambda_priv"],
    similarity_basis=s3["similarity_basis"], attacker_device=s3.get("attacker_device"),
    encoder_seeds=encoder_seeds, attacker_seeds=s3["attacker_seeds"],
    max_epochs=s3["max_epochs"], patience=s3["patience"], val_frac=s3["val_frac"], min_epochs=s3["min_epochs"],
    actual_epochs_per_encoder_seed=actual_epochs_per_encoder_seed,
    utility_mean=float(np.mean(utility_per_encoder_seed)), utility_std=float(np.std(utility_per_encoder_seed)),
    utility_per_encoder_seed=utility_per_encoder_seed,
    train_wall_time_seconds_per_encoder_seed=train_wall_time,
    n_users_before_filter=len(df), n_users_after_filter=len(df), n_excluded=0,
    chance_level=s3.get("chance_level"),
    raw_decile_ci=raw_ci, post_decile_ci=post_ci, inmodel_confidence_decile_ci=inmodel_ci,
    fairness_rescale=s3["fairness_rescale"],
    gradient_log_seed0=s3.get("gradient_log_seed0"),
    pooled_from=[str(dir_3seed), str(dir_7seed)],
)
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

protocol_tag = dict(
    n_encoder_seeds=10, protocol_ok=True, fairness_rescale=s3["fairness_rescale"], usable=True,
    usable_note="pooled from a 3-seed run (seeds 0-2) and a 7-seed run (seeds 3-9), same protocol/rescale",
    pooled_from=[str(dir_3seed), str(dir_7seed)],
)
(out_dir / "protocol_tag.json").write_text(json.dumps(protocol_tag, indent=2))

print(f"Pooled lambda4={l4}: {len(df)} users, 10 encoder seeds, utility_mean={summary['utility_mean']:.4f} "
      f"-> {out_dir}")
