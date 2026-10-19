import sys, time
sys.path.insert(0, "/data/mashhadi/puffin2")
import pandas as pd
import torch
from puffin.attacks.tul_maintul import run_tul_maintul, build_trajectory_dataset

df = pd.read_parquet("/data/mashhadi/puffin2/data/processed/geolife/unified.parquet")
device = torch.device("cuda:0")

t0 = time.time()
data = build_trajectory_dataset(df, max_traj_len=150, min_traj_len=3)
print(f"n_users (non-disjoint, >=1 train traj): {len(data['user_to_idx'])}")
print(f"total train trajs: {sum(len(v) for v in data['train'].values())}")
print(f"total test trajs: {sum(len(v) for v in data['test'].values())}")
print(f"total_locations: {data['total_locations']}")
print(f"built in {time.time()-t0:.1f}s")

t0 = time.time()
per_user_df, summary = run_tul_maintul(
    df, device, max_traj_len=150, min_traj_len=3, long_term_num=8,
    embed_size=128, hidden_size=256, num_layers=2, num_heads=4,
    batch_size=128, max_epochs=25, patience=4, lr=1e-3,
    temperature=10.0, lambda_distill=10.0, val_frac=0.1, seed=0,
)
print(f"trained+evaluated in {time.time()-t0:.1f}s")
print(summary)
print(per_user_df.describe())
per_user_df.to_csv("/tmp/tul_geolife_per_user.csv", index=False)
print("wrote /tmp/tul_geolife_per_user.csv")
