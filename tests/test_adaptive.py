#!/usr/bin/env python
"""Correctness checks for puffin/attacks/adaptive.py."""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.reid import GRUEncoder
from puffin.attacks.adaptive import (
    user_decile_map, per_group_specialized_attack, group_reweighted_attack,
    per_group_specialized_attack_multiseed, group_reweighted_attack_multiseed,
    can_infer_decile_from_features,
)

DEVICE = torch.device("cpu")


def _synthetic_windows(n_users=20, windows_per_user=20, sl=10, total_locations=50):
    """Each user has a DISTINCT favorite location (user_id itself, capped to
    total_locations) that dominates their windows -- an easy, well-separated
    synthetic re-id task so attacker correctness is checkable, not just
    "did it run"."""
    rng = np.random.default_rng(0)

    def build(n_per_user):
        cell, wd, hr, user = [], [], [], []
        for u in range(n_users):
            fav = (u % total_locations) + 1
            for _ in range(n_per_user):
                seq = rng.integers(1, total_locations + 1, size=sl)
                seq[rng.integers(0, sl)] = fav  # plant the signal
                cell.append(seq)
                wd.append(rng.integers(0, 7, size=sl))
                hr.append(rng.integers(0, 24, size=sl))
                user.append(u)
        return dict(cell=np.array(cell), weekday=np.array(wd), hour=np.array(hr),
                    user=np.array(user), next_cell=np.zeros(len(user), dtype=np.int64))

    return dict(train=build(windows_per_user), test=build(max(1, windows_per_user // 4)))


def _synthetic_feats(n_users=20):
    # s_unc spread evenly so qcut gives clean deciles.
    return pd.DataFrame(dict(user_id=np.arange(1, n_users + 1), s_unc=np.linspace(0, 10, n_users)))


def test_user_decile_map_spans_expected_range():
    feats = _synthetic_feats(20)
    dmap = user_decile_map(feats, n_deciles=10)
    assert set(dmap.keys()) == set(range(1, 21))
    assert set(dmap.values()) == set(range(10))  # 20 users / 10 deciles = clean 2-per-decile
    print("OK: user_decile_map covers all users across all deciles")


def test_per_group_specialized_attack_runs_and_labels_deciles_correctly():
    windows = _synthetic_windows(n_users=20, windows_per_user=15)
    feats = _synthetic_feats(20)
    encoder = GRUEncoder(context=10, total_locations=50).to(DEVICE)
    out = per_group_specialized_attack(encoder, windows, feats, total_users=20, device=DEVICE,
                                        n_deciles=10, epochs=3, batch_size=64)
    assert set(out.columns) >= {"user_id", "specialized_top1", "specialized_top5", "entropy_decile"}
    dmap = user_decile_map(feats, 10)
    for _, row in out.iterrows():
        assert dmap[row["user_id"]] == row["entropy_decile"]
    print(f"OK: per_group_specialized_attack ran on {len(out)} users, deciles correctly labeled")


def test_group_reweighted_attack_runs_and_labels_deciles_correctly():
    windows = _synthetic_windows(n_users=20, windows_per_user=15)
    feats = _synthetic_feats(20)
    encoder = GRUEncoder(context=10, total_locations=50).to(DEVICE)
    out = group_reweighted_attack(encoder, windows, feats, total_users=20, device=DEVICE,
                                   target_deciles=[0, 1], reweight_factor=5.0, epochs=3, batch_size=64)
    assert set(out.columns) >= {"user_id", "reweighted_top1", "reweighted_top5", "entropy_decile"}
    assert len(out) == 20  # all users evaluated, not just the target group
    print(f"OK: group_reweighted_attack ran on {len(out)} users")


def test_reweighting_shifts_accuracy_toward_target_group():
    """With a strong enough reweight_factor on a SMALL target group, that
    group's accuracy under the reweighted attacker should be at least as
    good as under an unweighted (reweight_factor=1) attacker trained the
    same way -- the whole point of variant (b)."""
    windows = _synthetic_windows(n_users=20, windows_per_user=25)
    feats = _synthetic_feats(20)
    target = [0]  # just decile 0 (2 users, given 20 users / 10 deciles)

    torch.manual_seed(0)
    encoder = GRUEncoder(context=10, total_locations=50).to(DEVICE)
    unweighted = group_reweighted_attack(encoder, windows, feats, total_users=20, device=DEVICE,
                                          target_deciles=target, reweight_factor=1.0, epochs=15,
                                          batch_size=64, seed=0)
    torch.manual_seed(0)
    encoder2 = GRUEncoder(context=10, total_locations=50).to(DEVICE)
    reweighted = group_reweighted_attack(encoder2, windows, feats, total_users=20, device=DEVICE,
                                          target_deciles=target, reweight_factor=20.0, epochs=15,
                                          batch_size=64, seed=0)

    dmap = user_decile_map(feats, 10)
    target_users = [u for u, d in dmap.items() if d in target]
    unw_acc = unweighted[unweighted["user_id"].isin(target_users)]["reweighted_top1"].mean()
    rew_acc = reweighted[reweighted["user_id"].isin(target_users)]["reweighted_top1"].mean()
    print(f"target-group accuracy: unweighted={unw_acc:.3f} reweighted(x20)={rew_acc:.3f}")
    assert rew_acc >= unw_acc - 1e-9, "heavy reweighting should not hurt the target group's accuracy"
    print("OK: reweighting toward the target group does not decrease its accuracy")


def test_multiseed_wrappers_aggregate_correctly():
    windows = _synthetic_windows(n_users=20, windows_per_user=15)
    feats = _synthetic_feats(20)
    encoder = GRUEncoder(context=10, total_locations=50).to(DEVICE)
    out = per_group_specialized_attack_multiseed(encoder, windows, feats, total_users=20, device=DEVICE,
                                                  seeds=(0, 1), epochs=3, batch_size=64)
    seed_cols = [c for c in out.columns if c.startswith("specialized_top1_seed")]
    assert len(seed_cols) == 2
    assert np.allclose(out["specialized_top1"], out[seed_cols].mean(axis=1))
    assert "specialized_top1_std" in out.columns

    encoder2 = GRUEncoder(context=10, total_locations=50).to(DEVICE)
    out2 = group_reweighted_attack_multiseed(encoder2, windows, feats, total_users=20, device=DEVICE,
                                              target_deciles=[0], seeds=(0, 1), epochs=3, batch_size=64)
    seed_cols2 = [c for c in out2.columns if c.startswith("reweighted_top1_seed")]
    assert len(seed_cols2) == 2
    assert np.allclose(out2["reweighted_top1"], out2[seed_cols2].mean(axis=1))
    print("OK: multiseed wrappers average correctly across seeds and keep per-seed columns")


def test_can_infer_decile_from_features_beats_chance_on_informative_signal():
    """Build windows where the LOCATION VOCABULARY itself leaks the decile
    (low-decile users confined to a small sub-range of locations, high-decile
    users spread across the full range) -- an encoder with real structure
    should recover decile membership well above chance (1/10)."""
    rng = np.random.default_rng(1)
    n_users, windows_per_user, sl, total_locations = 20, 30, 10, 100

    def build(n_per_user):
        cell, wd, hr, user = [], [], [], []
        for u in range(n_users):
            decile = u % 10
            # decile 0 -> locations confined to [1, 10]; decile 9 -> full range [1, 100]
            hi = 10 + decile * 10
            for _ in range(n_per_user):
                cell.append(rng.integers(1, hi + 1, size=sl))
                wd.append(rng.integers(0, 7, size=sl))
                hr.append(rng.integers(0, 24, size=sl))
                user.append(u)
        return dict(cell=np.array(cell), weekday=np.array(wd), hour=np.array(hr), user=np.array(user),
                    next_cell=np.zeros(len(user), dtype=np.int64))

    windows = dict(train=build(windows_per_user), test=build(max(1, windows_per_user // 4)))
    feats = _synthetic_feats(n_users)
    encoder = GRUEncoder(context=sl, total_locations=total_locations).to(DEVICE)
    out = can_infer_decile_from_features(encoder, windows, feats, device=DEVICE, epochs=15, batch_size=64)
    assert out["chance_accuracy"] == 0.1
    print(f"OK: can_infer_decile_from_features on an informative signal: {out} (chance=0.10)")
    assert out["user_level_majority_accuracy"] > out["chance_accuracy"]


if __name__ == "__main__":
    test_user_decile_map_spans_expected_range()
    test_per_group_specialized_attack_runs_and_labels_deciles_correctly()
    test_group_reweighted_attack_runs_and_labels_deciles_correctly()
    test_reweighting_shifts_accuracy_toward_target_group()
    test_multiseed_wrappers_aggregate_correctly()
    test_can_infer_decile_from_features_beats_chance_on_informative_signal()
    print("All adaptive-adversary tests passed.")
