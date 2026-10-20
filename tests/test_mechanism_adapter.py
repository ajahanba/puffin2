#!/usr/bin/env python
"""Correctness checks for puffin/attacks/mechanism_adapter.py (Phase 5b)."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.mechanism_adapter import train_encoder_on_released_windows
from puffin.attacks.adaptive import per_group_specialized_attack
from puffin.features.mobility import compute_mobility_features
import pandas as pd

DEVICE = torch.device("cpu")


def _synthetic_windows(n_users=20, windows_per_user=20, sl=10, total_locations=50):
    rng = np.random.default_rng(0)

    def build(n_per_user):
        cell, wd, hr, user = [], [], [], []
        for u in range(n_users):
            fav = (u % total_locations) + 1
            for _ in range(n_per_user):
                seq = rng.integers(1, total_locations + 1, size=sl)
                seq[rng.integers(0, sl)] = fav
                cell.append(seq)
                wd.append(rng.integers(0, 7, size=sl))
                hr.append(rng.integers(0, 24, size=sl))
                user.append(u)
        return dict(cell=np.array(cell), weekday=np.array(wd), hour=np.array(hr),
                    user=np.array(user), next_cell=np.zeros(len(user), dtype=np.int64))

    return dict(train=build(windows_per_user), test=build(max(1, windows_per_user // 4)))


def test_encoder_trains_and_separates_users_above_chance():
    """On the planted-favorite-location synthetic task (same construction
    as test_adaptive.py's), a classifier built on top of the trained
    encoder should clearly beat chance (1/20) -- confirms the encoder
    actually learns something usable, not just that it runs."""
    windows = _synthetic_windows(n_users=20, windows_per_user=25, total_locations=50)
    encoder = train_encoder_on_released_windows(windows, total_locations=50, total_users=20,
                                                 device=DEVICE, epochs=15, batch_size=64, seed=0)

    te = windows["test"]
    cell = torch.tensor(te["cell"], dtype=torch.long)
    wd = torch.tensor(te["weekday"], dtype=torch.long)
    hr = torch.tensor(te["hour"], dtype=torch.long)
    with torch.no_grad():
        feats = encoder(cell, wd, hr)
    assert feats.shape == (len(cell), 256)
    # a cheap linear probe on the frozen features should beat chance easily
    import torch.nn as nn
    probe = nn.Linear(256, 20)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    user_idx = torch.tensor(te["user"], dtype=torch.long)
    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(feats), user_idx)
        loss.backward()
        opt.step()
    acc = (probe(feats).argmax(-1) == user_idx).float().mean().item()
    print(f"OK: linear probe on trained encoder features gets {acc:.3f} (chance=0.05)")
    assert acc > 0.3


def test_encoder_is_compatible_with_adaptive_attack_functions():
    """The whole point: an encoder trained on 'released' data should slot
    directly into per_group_specialized_attack without any changes there."""
    windows = _synthetic_windows(n_users=20, windows_per_user=20, total_locations=50)
    encoder = train_encoder_on_released_windows(windows, total_locations=50, total_users=20,
                                                 device=DEVICE, epochs=10, batch_size=64, seed=0)
    feats = pd.DataFrame(dict(user_id=np.arange(1, 21), s_unc=np.linspace(0, 10, 20)))
    out = per_group_specialized_attack(encoder, windows, feats, total_users=20, device=DEVICE,
                                        n_deciles=10, epochs=3, batch_size=64)
    assert set(out.columns) >= {"user_id", "specialized_top1", "entropy_decile"}
    assert len(out) == 20
    print(f"OK: mechanism-adapter encoder works as a drop-in frozen encoder for "
          f"per_group_specialized_attack ({len(out)} users scored)")


if __name__ == "__main__":
    test_encoder_trains_and_separates_users_above_chance()
    test_encoder_is_compatible_with_adaptive_attack_functions()
    print("All mechanism_adapter tests passed.")
