#!/usr/bin/env python
"""Item 1, condition 3 (PI instruction, 2026-11-09): explore GeoLife's
TrajGAN loss curve over a large epoch budget (cheap now, ~40s/epoch) to
pick a fixed epoch budget where d_loss/g_loss have plateaued, not
diverged. Saves the raw curve to JSON for the same budget to be reused
on Foursquare/Gowalla-SF-Bay."""
import json
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, ".")
from puffin.data.tessellation import assign_grid_cells
from puffin.baselines.trajgan import run_trajgan

DATASET = sys.argv[1] if len(sys.argv) > 1 else "geolife"
PARQUET = f"data/processed/{DATASET}/unified.parquet"
MAX_EPOCHS = 150

df = pd.read_parquet(PARQUET)
df = df[df["lat"].notna() & df["lon"].notna()].copy()
df = assign_grid_cells(df, resolution_deg=0.01)

t0 = time.time()
_, summary = run_trajgan(
    df, torch.device("cuda:0"), max_traj_len=150, min_traj_len=11,
    max_epochs=MAX_EPOCHS, seed=0,
    cache_dir="/tmp/_trajgan_groups_cache", cache_key=DATASET)
elapsed = time.time() - t0

out = dict(
    dataset=DATASET, max_epochs=MAX_EPOCHS, elapsed_seconds=elapsed,
    d_loss_history=summary["d_loss_history"], g_loss_history=summary["g_loss_history"],
    epoch_times_seconds=summary["epoch_times_seconds"],
    n_train_chunks=summary["n_train_chunks"], n_test_chunks=summary["n_test_chunks"],
)
out_path = f"reports/figures/paper/trajgan_loss_curve_{DATASET}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {out_path}, elapsed={elapsed:.1f}s, {len(out['d_loss_history'])} epochs")
print("last 10 d_loss:", [round(x, 4) for x in out["d_loss_history"][-10:]])
print("last 10 g_loss:", [round(x, 4) for x in out["g_loss_history"][-10:]])
