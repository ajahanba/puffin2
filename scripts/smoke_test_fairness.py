#!/usr/bin/env python
"""Quick check that ed_soft runs end-to-end inside fixed_train_step without
crashing and actually produces a nonzero, backprop-able fairness loss. Not a
real experiment -- Phase 4 will own the real runs."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.data.geolife import load_geolife_sample, fixed_windows, compute_user_entropy
from puffin.models.mopae_torch import build_bundle, fixed_train_step
from puffin.fairness.losses import ed_soft, ed_exact, build_entropy_table

df = load_geolife_sample("data/raw/geolife_sample/GeoLifeBeijing_grid_train.csv",
                          "data/raw/geolife_sample/GeoLifeBeijing_grid_test.csv")
entropy_by_user = compute_user_entropy(df)
context = 10
total_locations = df["loc"].nunique()
total_users = df["label"].nunique()
(Xtr, ytr, ztr, wdtr, hrtr, ctr), _ = fixed_windows(df, context, total_locations, total_users)

device = torch.device("cpu")
bundle = build_bundle("fixed", context, total_locations, total_users)
for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
    m.to(device)
opts = dict(
    enc=torch.optim.Adam(bundle.encoder.parameters(), lr=3e-4),
    dec=torch.optim.Adam(bundle.decoder.parameters(), lr=3e-4),
    id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=3e-4),
    next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=3e-4),
)
entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

n = 256
loc = torch.tensor(Xtr[:n], dtype=torch.long)
wd = torch.tensor(wdtr[:n], dtype=torch.long)
hr = torch.tensor(hrtr[:n], dtype=torch.long)
y_idx = torch.tensor(ytr[:n].argmax(-1), dtype=torch.long)
z_idx = torch.tensor(ztr[:n].argmax(-1), dtype=torch.long)
coord = torch.tensor(ctr[:n], dtype=torch.float32)

# sanity: ed_exact should be ~inert (near-zero gradient contribution), ed_soft should not be.
enc = bundle.encoder(loc, wd, hr)
id_logits = bundle.id_recognizer(enc)
loss_exact = ed_exact(id_logits, z_idx, entropy_table)
loss_soft = ed_soft(id_logits, z_idx, entropy_table)
print("ed_exact (no grad path expected):", loss_exact.item(), "requires_grad:", loss_exact.requires_grad)
print("ed_soft  (should have a grad path):", loss_soft.item(), "requires_grad:", loss_soft.requires_grad)
grad_check = torch.autograd.grad(loss_soft, list(bundle.id_recognizer.parameters()), retain_graph=True, allow_unused=True)
print("ed_soft grad wrt id_recognizer (first param) is None:", grad_check[0] is None)

for step in range(5):
    out = fixed_train_step(bundle, opts, loc, wd, hr, y_idx, z_idx, coord,
                            fairness_fn=ed_soft, entropy_table=entropy_table, lambda4=0.1)
    print(step, out)
print("OK: fixed_train_step with ed_soft fairness term ran for 5 steps without error.")
