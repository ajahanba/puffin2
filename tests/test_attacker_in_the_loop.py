#!/usr/bin/env python
"""Correctness checks for puffin/fairness/attacker_in_the_loop.py."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.fairness.attacker_in_the_loop import refit_fresh_attacker_head, attacker_in_the_loop_train_step
from puffin.fairness.losses import floor_loss, group_dro_loss, build_entropy_table
from puffin.models.mopae_torch import build_bundle

DEVICE = torch.device("cpu")


def test_refit_fresh_attacker_head_beats_chance():
    torch.manual_seed(0)
    total_users, sl, total_locations = 20, 10, 50
    encoder = build_bundle("fixed", sl, total_locations, total_users).encoder
    n = 400
    cell = torch.randint(1, total_locations + 1, (n, sl))
    wd = torch.randint(0, 7, (n, sl))
    hr = torch.randint(0, 24, (n, sl))
    # plant a strong per-user signal in a distinctive favorite-location sense
    user_idx = torch.randint(0, total_users, (n,))
    for i in range(n):
        cell[i, 0] = (user_idx[i].item() % total_locations) + 1
    head = refit_fresh_attacker_head(encoder, cell, wd, hr, user_idx, total_users, DEVICE, epochs=15)
    with torch.no_grad():
        feat = encoder(cell, wd, hr)
        acc = (head(feat).argmax(-1) == user_idx).float().mean().item()
    assert acc > 1.0 / total_users
    print(f"OK: refit_fresh_attacker_head reaches {acc:.3f} accuracy on a planted signal (chance={1/total_users:.3f})")


def test_train_step_updates_encoder_not_fresh_head():
    torch.manual_seed(0)
    total_users, sl, total_locations = 20, 10, 50
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=1e-3),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=1e-3),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=1e-3),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=1e-3),
    )
    entropy_table = torch.linspace(0, 10, total_users)
    batch = 32
    loc = torch.randint(1, total_locations + 1, (batch, sl))
    weekday = torch.randint(0, 7, (batch, sl))
    hour = torch.randint(0, 24, (batch, sl))
    y_next_idx = torch.randint(0, total_locations, (batch,))
    z_user_idx = torch.randint(0, total_users, (batch,))
    coord_target = torch.zeros(batch, sl, 2)

    with torch.no_grad():
        feat = bundle.encoder(loc, weekday, hour)
    fresh_head = refit_fresh_attacker_head(bundle.encoder, loc, weekday, hour, z_user_idx, total_users,
                                            DEVICE, epochs=2)
    fresh_head_before = [p.clone() for p in fresh_head.parameters()]
    encoder_before = bundle.encoder.loc_emb.weight.clone()

    attacker_in_the_loop_train_step(bundle, opts, fresh_head, loc, weekday, hour, y_next_idx, z_user_idx,
                                     coord_target, floor_loss, entropy_table, lambda4=1.0)

    assert not torch.allclose(encoder_before, bundle.encoder.loc_emb.weight), "encoder must update"
    for before, after in zip(fresh_head_before, fresh_head.parameters()):
        assert torch.allclose(before, after), "fresh_head must NOT update during the main train step"
    print("OK: attacker_in_the_loop_train_step updates the encoder but leaves fresh_head frozen")


def test_works_with_both_loss_variants():
    torch.manual_seed(0)
    total_users, sl, total_locations = 20, 10, 50
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=1e-3),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=1e-3),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=1e-3),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=1e-3),
    )
    entropy_table = torch.linspace(0, 10, total_users)
    batch = 32
    loc = torch.randint(1, total_locations + 1, (batch, sl))
    weekday = torch.randint(0, 7, (batch, sl))
    hour = torch.randint(0, 24, (batch, sl))
    y_next_idx = torch.randint(0, total_locations, (batch,))
    z_user_idx = torch.randint(0, total_users, (batch,))
    coord_target = torch.zeros(batch, sl, 2)
    fresh_head = refit_fresh_attacker_head(bundle.encoder, loc, weekday, hour, z_user_idx, total_users,
                                            DEVICE, epochs=2)
    for loss_fn in (floor_loss, group_dro_loss):
        out = attacker_in_the_loop_train_step(bundle, opts, fresh_head, loc, weekday, hour, y_next_idx,
                                               z_user_idx, coord_target, loss_fn, entropy_table, lambda4=1.0)
        assert "loss_fair" in out
    print("OK: both floor_loss and group_dro_loss work as the attacker-in-the-loop fairness term")


if __name__ == "__main__":
    test_refit_fresh_attacker_head_beats_chance()
    test_train_step_updates_encoder_not_fresh_head()
    test_works_with_both_loss_variants()
    print("All attacker_in_the_loop tests passed.")
