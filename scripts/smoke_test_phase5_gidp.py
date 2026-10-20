import sys, time
sys.path.insert(0, "/data/mashhadi/puffin2")
import numpy as np
import pandas as pd
from puffin.baselines.gi_dp import run_gi_dp

df = pd.read_parquet("/data/mashhadi/puffin2/data/processed/geolife/unified.parquet")
print(f"n_rows={len(df)} n_users={df.user_id.nunique()}")

for eps in (0.001, 0.01, 0.1):
    t0 = time.time()
    out = run_gi_dp(df, epsilon=eps, seed=0)
    elapsed = time.time() - t0
    dlat_m = (out["lat"].to_numpy() - df["lat"].to_numpy()) * 111320.0
    dlon_m = (out["lon"].to_numpy() - df["lon"].to_numpy()) * 111320.0 * np.cos(np.radians(df["lat"].to_numpy()))
    dist = np.sqrt(dlat_m**2 + dlon_m**2)
    print(f"eps={eps}: ran in {elapsed:.1f}s, mean perturbation distance={dist.mean():.1f}m "
          f"(theoretical 2/eps={2/eps:.1f}m), median={np.median(dist):.1f}m, "
          f"n_unique_cell_id={out['cell_id'].nunique()} (real cell_id vocab was {df['cell_id'].nunique()})")

out01 = run_gi_dp(df, epsilon=0.01, seed=0)
out_path = "/tmp/gidp_synthetic_geolife_eps0.01.parquet"
out01.to_parquet(out_path)
print(f"wrote {out_path}")
print(out01.head())
