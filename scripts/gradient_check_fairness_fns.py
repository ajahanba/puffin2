#!/usr/bin/env python
"""Per-batch encoder-gradient norms for lipschitz_soft (entropy),
lipschitz_soft_pairtable (location_distribution), and group_dro_loss,
alongside ed_soft as reference -- same procedure as
scripts/gradient_check_location_puffin.py (GeoLife, 10 batches,
batch_size=512, lambda4=1.0, 1 seed, untrained fresh encoder, no
training). Also reports the location Lipschitz run's kernel-weight
(w_ij) distribution at sigma=1.
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.fairness.losses import (
    build_entropy_table, build_pairwise_distance_table,
    ed_soft, lipschitz_soft, lipschitz_soft_pairtable, group_dro_loss,
)
from puffin.models.mopae_torch import build_bundle
import puffin.metrics.individual_fairness as ifm

DEVICE = torch.device("cpu")
LAMBDA4 = 1.0
N_BATCHES = 10
BATCH_SIZE = 512
SEED = 0
SIGMA = 1.0

torch.manual_seed(SEED)
df = pd.read_parquet("data/processed/geolife/unified.parquet")
total_locations = int(df["cell_id"].max())
total_users = int(df["user_id"].max())
feats = compute_mobility_features(df)
entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
entropy_table = build_entropy_table(entropy_by_user, total_users, device=DEVICE)

train_df = ifm.train_only(df)
hist_by_user = ifm.per_user_cell_histogram(train_df, total_locations)
user_ids = sorted(hist_by_user.keys())
sim_matrix = ifm.sim_location_distribution(hist_by_user, user_ids)
pairwise_distance_table = build_pairwise_distance_table(sim_matrix, user_ids, total_users, device=DEVICE)

windows = build_sl_windows(df, sl=10, max_windows_per_user=200, respect_traj_boundary=False)
tr = windows["train"]
n = len(tr["user"])
rng = np.random.default_rng(SEED)

bundle = build_bundle("fixed", tr["cell"].shape[1], total_locations, total_users)
for m in [bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor]:
    m.to(DEVICE)


def encoder_grad_norm(loss, retain_graph=True):
    grads = torch.autograd.grad(loss, list(bundle.encoder.parameters()), retain_graph=retain_graph,
                                 allow_unused=True)
    flat = torch.cat([g.flatten() for g in grads if g is not None])
    return float(flat.norm().item())


def run_one_config(fairness_fn, fairness_table, label, collect_w=False):
    print(f"\n=== {label} (lambda4={LAMBDA4}) ===", flush=True)
    norms = {"recon": [], "util": [], "priv": [], "fair": []}
    w_samples = [] if collect_w else None
    for b in range(N_BATCHES):
        idx = rng.choice(n, size=BATCH_SIZE, replace=False)
        cell = torch.tensor(tr["cell"][idx], dtype=torch.long, device=DEVICE)
        wd = torch.tensor(tr["weekday"][idx], dtype=torch.long, device=DEVICE)
        hr = torch.tensor(tr["hour"][idx], dtype=torch.long, device=DEVICE)
        y_next = torch.tensor(tr["next_cell"][idx], dtype=torch.long, device=DEVICE)
        z_user = torch.tensor(tr["user"][idx], dtype=torch.long, device=DEVICE)
        coord_target = torch.zeros((len(cell), cell.shape[1], 2), dtype=torch.float32, device=DEVICE)

        enc = bundle.encoder(cell, wd, hr)
        recon = bundle.decoder(enc)
        loss_recon = F.mse_loss(recon, coord_target)
        next_logits = bundle.next_predictor(enc)
        loss_util = F.cross_entropy(next_logits, y_next)
        id_logits = bundle.id_recognizer(enc)
        loss_priv = F.cross_entropy(id_logits, z_user)
        loss_fair = fairness_fn(id_logits, z_user, fairness_table)

        norms["recon"].append(encoder_grad_norm(loss_recon))
        norms["util"].append(encoder_grad_norm(loss_util))
        norms["priv"].append(encoder_grad_norm(loss_priv))
        norms["fair"].append(encoder_grad_norm(loss_fair, retain_graph=False))

        if collect_w:
            with torch.no_grad():
                d = pairwise_distance_table[z_user][:, z_user]
                w = torch.exp(-(d ** 2) / (2 * SIGMA ** 2))
                off_diag = w[~torch.eye(w.shape[0], dtype=torch.bool)]
                w_samples.append(off_diag.flatten().numpy())

    for k, v in norms.items():
        print(f"  {k}: mean={np.mean(v):.6e}  (min {np.min(v):.3e}, max {np.max(v):.3e})")
    if collect_w:
        all_w = np.concatenate(w_samples)
        print(f"  w_ij (sigma={SIGMA}): median={np.median(all_w):.6e}  "
              f"p10={np.percentile(all_w, 10):.6e}  p90={np.percentile(all_w, 90):.6e}  n={len(all_w)}")
        return norms, all_w
    return norms, None


results = {}
configs = [
    ("ed_soft (entropy)", ed_soft, entropy_table, False),
    ("lipschitz_soft (entropy)", lipschitz_soft, entropy_table, False),
    ("lipschitz_soft_pairtable (location_distribution)", lipschitz_soft_pairtable, pairwise_distance_table, True),
    ("group_dro_loss (entropy)", group_dro_loss, entropy_table, False),
]
for label, fn, table, collect_w in configs:
    norms, w = run_one_config(fn, table, label, collect_w=collect_w)
    results[label] = norms

ref_fair_mean = np.mean(results["ed_soft (entropy)"]["fair"])
print(f"\n=== Summary: fairness-term encoder-gradient norm, mean over {N_BATCHES} batches ===")
print(f"{'label':55s} {'fair_norm':>14s} {'ratio_to_ed_soft':>18s}")
for label, _, _, _ in configs:
    v = np.mean(results[label]["fair"])
    print(f"{label:55s} {v:14.6e} {v / ref_fair_mean:18.6e}")
