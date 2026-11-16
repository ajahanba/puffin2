#!/usr/bin/env python
"""Correctness check for train_attacker_on_frozen_encoder_strongest_per_user
(the frozen-encoder half of the strongest-per-user attacker standard)."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.reid import GRUEncoder, train_attacker_on_frozen_encoder_strongest_per_user

DEVICE = torch.device("cpu")


def _synthetic_windows(n_users=15, windows_per_user=25, sl=10, total_locations=40):
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
                    user=np.array(user), next_cell=np.zeros(len(user), dtype=np.int64),
                    ts=np.arange(len(user)))

    return dict(train=build(windows_per_user), test=build(max(1, windows_per_user // 4)))


def test_runs_and_beats_chance():
    windows = _synthetic_windows()
    encoder = GRUEncoder(context=10, total_locations=40)
    out, arch_win_log = train_attacker_on_frozen_encoder_strongest_per_user(
        encoder, windows, total_users=15, device=DEVICE, seeds=(0, 1), max_epochs=5, patience=3)
    assert set(out.columns) >= {"user_id", "post_top1", "post_top1_std"}
    assert len(out) == 15
    assert out["post_top1"].mean() > 1.0 / 15
    assert len(arch_win_log) == 2
    print(f"OK: frozen-encoder strongest-per-user attacker ran, mean post_top1="
          f"{out['post_top1'].mean():.3f} (chance=1/15={1/15:.3f}), win_fracs={arch_win_log}")


def test_per_user_max_is_at_least_mlp_alone():
    """The per-user max across {mlp, gbt} can never be less than mlp alone
    for any individual user -- a basic invariant."""
    windows = _synthetic_windows()
    encoder = GRUEncoder(context=10, total_locations=40)
    both, _ = train_attacker_on_frozen_encoder_strongest_per_user(
        encoder, windows, total_users=15, device=DEVICE, seeds=(0,), max_epochs=5, patience=3, include_gbt=True)
    mlp_only, _ = train_attacker_on_frozen_encoder_strongest_per_user(
        encoder, windows, total_users=15, device=DEVICE, seeds=(0,), max_epochs=5, patience=3, include_gbt=False)
    merged = both.merge(mlp_only, on="user_id", suffixes=("_both", "_mlp"))
    assert (merged["post_top1_both"] >= merged["post_top1_mlp"] - 1e-9).all()
    print("OK: per-user max across {mlp,gbt} is never less than mlp alone, for every user")


if __name__ == "__main__":
    test_runs_and_beats_chance()
    test_per_user_max_is_at_least_mlp_alone()
    print("All frozen-encoder strongest-per-user tests passed.")
