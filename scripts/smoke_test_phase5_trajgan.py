import sys, time, argparse
sys.path.insert(0, "/data/mashhadi/puffin2")
import pandas as pd
import torch
from puffin.baselines.trajgan import run_trajgan

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="geolife_sample")
ap.add_argument("--max_epochs", type=int, default=30)
ap.add_argument("--max_traj_len", type=int, default=100)
ap.add_argument("--batch_size", type=int, default=128)
args = ap.parse_args()

df = pd.read_parquet(f"/data/mashhadi/puffin2/data/processed/{args.dataset}/unified.parquet")
device = torch.device("cpu")

print(f"dataset={args.dataset} n_rows={len(df)} n_users={df.user_id.nunique()}")

t0 = time.time()
synthetic_df, summary = run_trajgan(
    df, device, max_traj_len=args.max_traj_len, min_traj_len=5,
    latent_dim=100, lstm_units=100, batch_size=args.batch_size,
    max_epochs=args.max_epochs, lr=1e-3, seed=0,
)
elapsed = time.time() - t0
print(f"trained+generated in {elapsed:.1f}s")
for k, v in summary.items():
    if k not in ("d_loss_history", "g_loss_history"):
        print(f"  {k}: {v}")
print("d_loss first/last 5:", summary["d_loss_history"][:5], "...", summary["d_loss_history"][-5:])
print("g_loss first/last 5:", summary["g_loss_history"][:5], "...", summary["g_loss_history"][-5:])
print(synthetic_df.head())
print(synthetic_df.describe())
out_path = f"/tmp/trajgan_synthetic_{args.dataset}.parquet"
synthetic_df.to_parquet(out_path)
print(f"wrote {out_path}")
