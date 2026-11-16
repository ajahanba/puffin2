#!/usr/bin/env python
"""Per-batch gradient norms on the
ENCODER from each loss term (reconstruction, URU=privacy/id-recognizer,
MPU=utility/next-location, fairness) for ed_soft (entropy) vs
ed_soft_location (location_distribution) at lambda4=1.0, GeoLife, 1 seed.
Measures each term's gradient on the encoder INDEPENDENTLY (separate
torch.autograd.grad calls, retain_graph=True) -- fixed_train_step's own
single combined enc_total gradient can't be decomposed after the fact."""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.fairness.losses import build_entropy_table, build_histogram_table, ed_soft, ed_soft_location
from puffin.models.mopae_torch import build_bundle, FIXED_LAMBDAS
import puffin.metrics.individual_fairness as ifm

DEVICE = torch.device("cpu")  # tiny diagnostic, fast regardless of device
LAMBDA4 = 1.0
N_BATCHES = 10
BATCH_SIZE = 512
SEED = 0

torch.manual_seed(SEED)
df = pd.read_parquet("data/processed/geolife/unified.parquet")
total_locations = int(df["cell_id"].max())
total_users = int(df["user_id"].max())
feats = compute_mobility_features(df)
entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
entropy_table = build_entropy_table(entropy_by_user, total_users, device=DEVICE)

train_df = ifm.train_only(df)
hist_by_user = ifm.per_user_cell_histogram(train_df, total_locations)
histogram_table = build_histogram_table(hist_by_user, total_users, total_locations, device=DEVICE)

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


def run_one_config(fairness_fn, fairness_table, label):
    print(f"\n=== {label} (lambda4={LAMBDA4}) ===")
    norms = {"recon": [], "util": [], "priv": [], "fair": []}
    for b in range(N_BATCHES):
        idx = rng.choice(n, size=BATCH_SIZE, replace=False)
        cell = torch.tensor(tr["cell"][idx], dtype=torch.long, device=DEVICE)
        wd = torch.tensor(tr["weekday"][idx], dtype=torch.long, device=DEVICE)
        hr = torch.tensor(tr["hour"][idx], dtype=torch.long, device=DEVICE)
        y_next = torch.tensor(tr["next_cell"][idx], dtype=torch.long, device=DEVICE)
        z_user = torch.tensor(tr["user"][idx], dtype=torch.long, device=DEVICE)
        # Matches train_mechanism.py's real call exactly:
        # coord_target is an all-zeros (B, sl, 2) tensor -- the project's actual convention,
        # not a placeholder of mine. Replicated as-is so the gradient magnitudes measured
        # here match the real training loop's, not an approximation of it.
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

    for k, v in norms.items():
        print(f"  {k}: mean={np.mean(v):.6e}  (min {np.min(v):.3e}, max {np.max(v):.3e})")
    return norms


norms_entropy = run_one_config(ed_soft, entropy_table, "ed_soft (entropy)")
norms_location = run_one_config(ed_soft_location, histogram_table, "ed_soft_location (location_distribution)")

ratio = np.mean(norms_entropy["fair"]) / np.mean(norms_location["fair"])
print(f"\nratio ed_soft_fair_norm / ed_soft_location_fair_norm = {ratio:.3e}")
print(f"Location term's norm is {'>=' if ratio >= 10 else '<'} 10x smaller than ed_soft's -> "
      f"{'RESCALE NEEDED' if ratio >= 10 else 'no rescale needed'}")
print(f"\nSuggested rescale_factor for ed_soft_location = {ratio:.6e} "
      f"(multiply ed_soft_location's returned loss by this before lambda4)")
