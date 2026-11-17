#!/usr/bin/env python
"""Same as compute_yjmob_missing_sims.py, pointed at the second
YJMob100K sample (yjmob100k_sample_s2), indexed by the remapped dense
1..2000 user ids that unified_sample.parquet and every yjmob100k_s2_*
run's per_user_04d.csv actually use."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm

BASE = pathlib.Path("reports/figures/yjmob100k_sample_s2")
df = pd.read_parquet(BASE / "unified_sample.parquet")
total_locations = int(df["cell_id"].max())
tr = ifm.train_only(df)
feats = ifm.compute_mobility_features(tr)
entropy_by_user = dict(zip(feats.user_id, feats.s_unc))
rog_by_user = dict(zip(feats.user_id, feats.rog_km))
hist_by_user = ifm.per_user_cell_histogram(tr, total_locations)
seq_by_user = ifm.per_user_latlon_sequence(tr, max_points=150)

user_ids = sorted(int(u) for u in df["user_id"].unique())
print(f"n user_ids (remapped)={len(user_ids)}, n with entropy={len(entropy_by_user)}, "
      f"n with rog={len(rog_by_user)}, n with hist={len(hist_by_user)}, n with seq={len(seq_by_user)}")

sims = {}
sims["entropy"] = ifm.sim_entropy(entropy_by_user, user_ids)
print("entropy done", flush=True)
sims["location_distribution"] = ifm.sim_location_distribution(hist_by_user, user_ids)
print("location_distribution done", flush=True)
sims["radius_of_gyration"] = ifm.sim_radius_of_gyration(rog_by_user, user_ids)
print("radius_of_gyration done", flush=True)
sims["trajectory_shape"] = ifm.sim_trajectory_shape(seq_by_user, user_ids)
print("trajectory_shape done", flush=True)

np.savez(BASE / "similarities.npz", **sims)
(BASE / "user_ids.json").write_text(json.dumps(user_ids))
print(f"wrote {BASE / 'similarities.npz'} and {BASE / 'user_ids.json'}, keys: {list(sims.keys())}")
