#!/usr/bin/env python
"""Diagnoses whether floor_loss is actually contributing gradient during
GeoLife training -- its lambda4=0.3/1.0/3.0 outputs were identical in
Checkpoint 4/04c, suggesting it may be inert (always zero) throughout
training, the same failure mode this project already found once in the
original PUFFIN `ed` loss."""
import pathlib
import sys

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.fairness.losses import floor_loss, build_entropy_table
from puffin.models.mopae_torch import build_bundle, fixed_train_step

df = pd.read_parquet("data/processed/geolife/unified.parquet")
total_locations = int(df["cell_id"].max())
total_users = int(df["user_id"].max())
feats = compute_mobility_features(df)
entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

windows = build_sl_windows(df, sl=10, max_windows_per_user=200, respect_traj_boundary=True)

torch.manual_seed(0)
sl = windows["train"]["cell"].shape[1]
bundle = build_bundle("fixed", sl, total_locations, total_users)
for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
    m.to(device)
opts = dict(
    enc=torch.optim.Adam(bundle.encoder.parameters(), lr=3e-4),
    dec=torch.optim.Adam(bundle.decoder.parameters(), lr=3e-4),
    id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=3e-4),
    next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=3e-4),
)
lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=0.3)

tr = windows["train"]
cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)
coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
n = len(cell)
batch_size = 512

print("epoch, step, floor_loss_value, confidence_max, confidence_mean, encoder_grad_norm_from_fair_term")
for epoch in range(5):
    perm = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        b_cell, b_wd, b_hr = cell[idx], wd[idx], hr[idx]
        b_y_next, b_user, b_coord = y_next_idx[idx], user_idx[idx], coord_target[idx]

        # manually replicate fixed_train_step's forward pass to inspect floor_loss directly
        enc = bundle.encoder(b_cell, b_wd, b_hr)
        id_logits = bundle.id_recognizer(enc)
        test_floor = 2.0 / total_users  # proposed fix: 2x chance level
        loss_fair = floor_loss(id_logits, b_user, entropy_table, floor=test_floor)

        import torch.nn.functional as F
        probs = F.softmax(id_logits, dim=-1)
        true_onehot = F.one_hot(b_user, num_classes=id_logits.shape[-1]).float()
        confidence = (true_onehot * probs).sum(-1)

        if start == 0:
            if loss_fair.requires_grad and enc.requires_grad:
                grad = torch.autograd.grad(loss_fair, list(bundle.encoder.parameters()),
                                            retain_graph=False, allow_unused=True)
                grad_norm = sum((g.norm().item() if g is not None else 0.0) for g in grad)
            else:
                grad_norm = float("nan")
            n_above_floor = (confidence > test_floor).sum().item()
            print(f"{epoch}, {start}, loss={loss_fair.item():.3e}, conf_max={confidence.max().item():.6f}, "
                  f"conf_mean={confidence.mean().item():.6f}, floor={test_floor:.6f}, "
                  f"n_above_floor={n_above_floor}/{len(confidence)}, grad_norm={grad_norm:.3e}")

        fixed_train_step(bundle, opts, b_cell, b_wd, b_hr, b_y_next, b_user, b_coord,
                          fairness_fn=floor_loss, entropy_table=entropy_table, lambda4=1.0, lambdas=lambdas)
