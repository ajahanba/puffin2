#!/usr/bin/env python
"""Quick check that train_raw_attacker_strongest_per_user (Phase 3 task 2's
finalized attacker: MLP + GRU + gradient-boosted trees, per-user max) runs
end-to-end and that the per-user max/winner logic is self-consistent."""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.data.splits import temporal_split
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import train_raw_attacker_strongest_per_user

df = pd.read_parquet("data/processed/geolife/unified.parquet")
total_locations = int(df["cell_id"].max())
total_users = int(df["user_id"].max())
windows = build_sl_windows(df, sl=10, max_windows_per_user=30, respect_traj_boundary=True)
print(f"train windows={len(windows['train']['user'])} test windows={len(windows['test']['user'])}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out, win_log = train_raw_attacker_strongest_per_user(
    windows, total_locations, total_users, device, seeds=(0, 1), max_epochs=4, patience=2)

print("arch_win_log:", win_log)
print(out.head())
print("columns:", list(out.columns))

# Sanity: the reported top1 for each seed must equal the max of the 3
# architectures' top1 for that user (can't check this directly here without
# re-running each architecture, so just check internal consistency: raw_top1
# is the mean across seed columns, and every seed's top1 is in [0,1]).
top1_cols = [c for c in out.columns if c.startswith("top1_seed")]
assert (out[top1_cols] >= 0).all().all() and (out[top1_cols] <= 1).all().all()
assert np.allclose(out["raw_top1"], out[top1_cols].mean(axis=1))
winner_cols = [c for c in out.columns if c.startswith("winner_seed")]
assert set(out[winner_cols[0]].unique()) <= {"mlp", "gru", "gbt"}
print("OK: strongest-per-user attacker ran end-to-end with self-consistent output")
