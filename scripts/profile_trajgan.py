#!/usr/bin/env python
"""Profile LSTM-TrajGAN's per-batch cost on GeoLife, per the PI's
instruction (2026-11-09). Reuses the cached trajectory groups so this
profile excludes the (separately slow, CPU-bound) grouping step."""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from puffin.baselines.trajgan import (
    TrajGANGenerator, TrajGANDiscriminator, _latlon_centroid_scale, _pad_batch, _traj_loss,
)

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 64
LATENT_DIM = 100
LSTM_UNITS = 100
MAX_TRAJ_LEN = 150
N_PROFILE_BATCHES = 30

with open("/tmp/_trajgan_groups_cache/trajgan_groups_geolife_150_11.pkl", "rb") as f:
    train_groups, test_groups = pickle.load(f)
print(f"n_train_trajectories={len(train_groups)}, n_test_trajectories={len(test_groups)}")
n_train = len(train_groups)
batches_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
print(f"batch_size={BATCH_SIZE}, batches_per_epoch={batches_per_epoch}")

lat_c, lon_c, scale = _latlon_centroid_scale(train_groups, test_groups)

torch.manual_seed(0)
G = TrajGANGenerator(LATENT_DIM, LSTM_UNITS).to(DEVICE)
D = TrajGANDiscriminator(LSTM_UNITS).to(DEVICE)
print(f"next(G.parameters()).device = {next(G.parameters()).device}")
print(f"next(D.parameters()).device = {next(D.parameters()).device}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}, "
      f"torch.cuda.current_device() = {torch.cuda.current_device() if torch.cuda.is_available() else None}")

opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))

rng = np.random.default_rng(0)
order = rng.permutation(n_train)

pad_times, fwd_bwd_times, total_times = [], [], []
n_done = 0
for start in range(0, min(N_PROFILE_BATCHES * BATCH_SIZE, n_train), BATCH_SIZE):
    idx = order[start:start + BATCH_SIZE]
    if len(idx) < 2:
        continue
    t_batch_start = time.time()

    t0 = time.time()
    latlon, weekday_oh, hour_oh, mask, lengths = _pad_batch(train_groups, idx, MAX_TRAJ_LEN, lat_c, lon_c, scale)
    latlon, weekday_oh, hour_oh, mask = (
        latlon.to(DEVICE), weekday_oh.to(DEVICE), hour_oh.to(DEVICE), mask.to(DEVICE))
    lengths_t = torch.from_numpy(lengths).to(DEVICE)
    torch.cuda.synchronize()
    pad_time = time.time() - t0
    pad_times.append(pad_time)

    B = len(idx)
    t0 = time.time()

    noise = torch.randn(B, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake_latlon, fake_weekday, fake_hour = G(latlon, weekday_oh, hour_oh, noise, scale)
    d_real = D(latlon, weekday_oh, hour_oh)
    d_fake = D(fake_latlon, fake_weekday, fake_hour)
    d_loss = 0.5 * (F.binary_cross_entropy(d_real, torch.ones_like(d_real)) +
                     F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    noise = torch.randn(B, LATENT_DIM, device=DEVICE)
    gen_latlon, gen_weekday, gen_hour = G(latlon, weekday_oh, hour_oh, noise, scale)
    d_pred_fake = D(gen_latlon, gen_weekday, gen_hour)
    g_loss = _traj_loss(mask, lengths_t, latlon, gen_latlon, weekday_oh, gen_weekday,
                         hour_oh, gen_hour, d_pred_fake, 1.0, 10.0, 1.0, 1.0)
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    torch.cuda.synchronize()
    fwd_bwd_time = time.time() - t0
    fwd_bwd_times.append(fwd_bwd_time)
    total_times.append(time.time() - t_batch_start)
    n_done += 1

print(f"\nProfiled {n_done} batches:")
print(f"  mean _pad_batch time:      {np.mean(pad_times)*1000:.2f} ms  (min {np.min(pad_times)*1000:.2f}, max {np.max(pad_times)*1000:.2f})")
print(f"  mean forward/backward time: {np.mean(fwd_bwd_times)*1000:.2f} ms  (min {np.min(fwd_bwd_times)*1000:.2f}, max {np.max(fwd_bwd_times)*1000:.2f})")
print(f"  mean total batch time:     {np.mean(total_times)*1000:.2f} ms")
print(f"\nExtrapolated: {batches_per_epoch} batches/epoch x {np.mean(total_times):.4f}s/batch "
      f"= {batches_per_epoch * np.mean(total_times):.1f}s/epoch "
      f"= {batches_per_epoch * np.mean(total_times)/60:.2f} min/epoch")
