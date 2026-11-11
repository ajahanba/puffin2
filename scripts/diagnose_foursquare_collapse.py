#!/usr/bin/env python
"""Foursquare TrajGAN collapse diagnosis (PI instruction, 2026-11-10):
collapse at epoch 2-4 suggests D is exploiting a trivial artifact, not
ordinary instability. Runs the PI's 4-part diagnostic on one seed before
applying the stop rule.
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from puffin.data.tessellation import assign_grid_cells
from puffin.baselines.trajgan import (
    _trajectory_chunks, _latlon_centroid_scale, _pad_batch,
    TrajGANGenerator, TrajGANDiscriminator,
)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda:1")

df = pd.read_parquet("data/processed/foursquare/unified.parquet")
df = df[df["lat"].notna() & df["lon"].notna()].copy()
df = assign_grid_cells(df, resolution_deg=0.01)

train_groups = _trajectory_chunks(df, "train", max_traj_len=150, min_traj_len=2, group_by="week")
test_groups = _trajectory_chunks(df, "test", max_traj_len=150, min_traj_len=1, group_by="week")
lat_c, lon_c, scale = _latlon_centroid_scale(train_groups, test_groups)
print(f"=== Part 2: coordinate scaling ===")
print(f"lat_centroid={lat_c:.6f}, lon_centroid={lon_c:.6f}, scale_factor={scale:.6f} (raw degrees)")
all_lat = np.concatenate([g["lat"] for g in train_groups])
all_lon = np.concatenate([g["lon"] for g in train_groups])
print(f"raw lat range=[{all_lat.min():.4f},{all_lat.max():.4f}], "
      f"raw lon range=[{all_lon.min():.4f},{all_lon.max():.4f}]")
print(f"deviation-from-centroid transform: (lat-lat_c)/scale, clipped to [-1,1] -- "
      f"SAME transform/code path used for GeoLife (puffin/baselines/trajgan.py:_latlon_centroid_scale, "
      f"_pad_batch; no dataset-specific branch exists).")
print(f"NOTE: scale_factor is a raw-degree deviation (city-scale Foursquare should be small, "
      f"this one is {scale:.6f} degrees) -- this value matters for part 1/3 below because the "
      f"GENERATOR'S OUTPUT uses it as a multiplier (see puffin/baselines/trajgan.py "
      f"TrajGANGenerator.forward: `gen_latlon = torch.tanh(...) * scale_factor`), while the REAL "
      f"latlon fed to D is normalized by DIVIDING by scale_factor in _pad_batch -- i.e. real in "
      f"[-1,1], candidate-bug: fake in [-scale_factor,scale_factor].")

rng = np.random.default_rng(0)
idx = rng.choice(len(train_groups), size=min(64, len(train_groups)), replace=False).tolist()
latlon, weekday_oh, hour_oh, mask, lengths = _pad_batch(train_groups, idx, 150, lat_c, lon_c, scale)
latlon, weekday_oh, hour_oh, mask = latlon.to(DEVICE), weekday_oh.to(DEVICE), hour_oh.to(DEVICE), mask.to(DEVICE)

G = TrajGANGenerator(latent_dim=100, lstm_units=100).to(DEVICE)
D = TrajGANDiscriminator(lstm_units=100).to(DEVICE)
noise = torch.randn(len(idx), 100, device=DEVICE)
with torch.no_grad():
    fake_latlon, fake_weekday, fake_hour = G(latlon, weekday_oh, hour_oh, noise, scale, mask)

print(f"\n=== Part 1: feature comparison at epoch 0 (untrained G), one batch of {len(idx)} ===")
mask_np = mask.cpu().numpy().astype(bool)
real_ll = latlon.cpu().numpy()
fake_ll = fake_latlon.cpu().numpy()
print(f"REAL  latlon: masked range=[{real_ll[mask_np.astype(bool).repeat(2,axis=-1).reshape(real_ll.shape)].min():.4f},"
      f"{real_ll[mask_np.astype(bool).repeat(2,axis=-1).reshape(real_ll.shape)].max():.4f}], "
      f"mean_abs={np.abs(real_ll[np.stack([mask_np,mask_np],axis=-1)]).mean():.4f}")
print(f"FAKE  latlon: range=[{fake_ll.min():.4f},{fake_ll.max():.4f}], mean_abs={np.abs(fake_ll).mean():.4f} "
      f"(includes padded positions -- fake has no mask applied, same as D's own input in training)")
print(f"REAL padding value at non-mask positions: {real_ll[~np.stack([mask_np,mask_np],axis=-1)].max():.6f} "
      f"(should be exactly 0.0, pre-padding convention)")
print(f"FAKE value at positions real-padding occupies (same pre-pad slots): "
      f"mean_abs={np.abs(fake_ll[~np.stack([mask_np,mask_np],axis=-1)]).mean():.4f} "
      f"(generator has NO mask input/output gating -- it freely emits non-zero values at padded "
      f"positions, same as the original architecture (D also receives unmasked fake AND unmasked "
      f"real-pre-padded-with-TRUE-zeros input, so both are zero there for real, non-zero for fake "
      f"pre-training -- but an UNTRAINED generator emitting nonzero at the padding slots IS a second, "
      f"independent trivial signal D can exploit alongside the scale mismatch)")
print(f"weekday one-hot sums (should be 1.0 at every position, real and fake): "
      f"real={weekday_oh.sum(-1).mean().item():.4f}, fake={fake_weekday.sum(-1).mean().item():.4f}")
print(f"hour one-hot sums: real={hour_oh.sum(-1).mean().item():.4f}, fake={fake_hour.sum(-1).mean().item():.4f}")

print(f"\n=== Part 3: discriminator-alone ablation, 1 epoch, real vs UNTRAINED-G fake ===")


def train_d_one_epoch(real_ll_in, fake_ll_in, real_wd_in, fake_wd_in, real_hr_in, fake_hr_in, label):
    D_local = TrajGANDiscriminator(lstm_units=100).to(DEVICE)
    opt = torch.optim.Adam(D_local.parameters(), lr=1e-3, betas=(0.5, 0.999))
    correct, total = 0, 0
    for _ in range(30):  # 30 steps ~ "1 epoch" over this batch-sized diagnostic sample
        d_real = D_local(real_ll_in, real_wd_in, real_hr_in)
        d_fake = D_local(fake_ll_in, fake_wd_in, fake_hr_in)
        loss = (F.binary_cross_entropy(d_real, torch.ones_like(d_real)) +
                F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d_real = D_local(real_ll_in, real_wd_in, real_hr_in)
        d_fake = D_local(fake_ll_in, fake_wd_in, fake_hr_in)
        correct = (d_real > 0.5).sum().item() + (d_fake < 0.5).sum().item()
        total = d_real.numel() + d_fake.numel()
    acc = correct / total
    print(f"  ({label}): D accuracy after 1-epoch-equivalent training = {acc:.4f}")
    return acc


zeros_ll = torch.zeros_like(latlon)
zeros_wd = torch.zeros_like(weekday_oh)
zeros_hr = torch.zeros_like(hour_oh)

train_d_one_epoch(latlon, fake_latlon, weekday_oh, fake_weekday, hour_oh, fake_hour, "a: all features")
train_d_one_epoch(latlon, fake_latlon, zeros_wd, zeros_wd, zeros_hr, zeros_hr, "b: coordinates only")
train_d_one_epoch(zeros_ll, zeros_ll, weekday_oh, fake_weekday, hour_oh, fake_hour, "c: time features only")
mask3 = mask.unsqueeze(-1)
real_ll_zm = latlon * mask3
fake_ll_zm = fake_latlon * mask3
real_wd_zm = weekday_oh * mask3
fake_wd_zm = fake_weekday * mask3
real_hr_zm = hour_oh * mask3
fake_hr_zm = fake_hour * mask3
train_d_one_epoch(real_ll_zm, fake_ll_zm, real_wd_zm, fake_wd_zm, real_hr_zm, fake_hr_zm,
                   "d: padding positions zeroed in both (post-fix: this is now the SAME as a/b/c "
                   "since mask=mask was passed to G above -- fake is already masked)")

print(f"\n=== Part 4: diff vs original LSTM-TrajGAN repo (from this project's own port-decision docstring, "
      f"puffin/baselines/trajgan.py module docstring) ===")
print("Documented, deliberate differences: category field dropped (no analogue in this project's "
      "schema), scale_factor uses 99.5th percentile not true max (GeoLife outlier robustness), "
      "max_traj_len truncation at 150, full-epoch training loop not original's random-batch loop.")
print("NOT documented anywhere as a deliberate decision, and not present in the original "
      "(`build_generator`'s lat_lon output is `Dense(2, activation='tanh')` with NO rescale -- "
      "the original keeps the generator's output in the SAME normalized [-1,1] space as the real "
      "input the whole time, only multiplying by scale_factor in predict.py's CSV-writing step, "
      "after training/adversarial comparison is done): this port's TrajGANGenerator.forward does "
      "`torch.tanh(self.out_latlon(lstm_out)) * scale_factor` INSIDE the network itself, so the "
      "fake latlon D compares against real latlon is in [-scale_factor,scale_factor], not [-1,1].")
