#!/usr/bin/env python
"""Correctness checks for Phase 4's new formulations (puffin/fairness/losses.py
additions + puffin/fairness/group_adversarial.py)."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.fairness.losses import (
    lipschitz_soft, floor_loss, variance_of_risk, group_dro_loss, per_user_minimax_loss,
    _per_sample_confidence, build_entropy_table,
)
from puffin.fairness.group_adversarial import GroupAdversaryHeads, group_adversarial_train_step, decile_group_idx
from puffin.models.mopae_torch import build_bundle

DEVICE = torch.device("cpu")


def _toy_logits_and_users(n_users=20, batch=40, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(batch, n_users)
    users = torch.randint(0, n_users, (batch,))
    entropy_table = torch.linspace(0, 10, n_users)
    return logits, users, entropy_table


def test_lipschitz_soft_zero_when_confidence_uniform():
    logits, users, entropy_table = _toy_logits_and_users()
    # force identical confidence for everyone regardless of entropy
    logits = torch.zeros_like(logits)  # softmax -> uniform -> identical confidence for all
    loss = lipschitz_soft(logits, users, entropy_table)
    assert loss.item() < 1e-6
    print(f"OK: lipschitz_soft ~0 ({loss.item():.2e}) when confidence is uniform across users")


def test_lipschitz_soft_penalizes_similar_entropy_mismatch_more_than_dissimilar():
    n_users = 4
    entropy_table = torch.tensor([0.0, 0.01, 5.0, 5.01])  # (0,1) close pair, (2,3) close pair, far from each other
    users = torch.tensor([0, 1, 2, 3])
    # confidence: user0 high, user1 low (close-entropy pair, mismatched) ;
    # user2 high, user3 high (close-entropy pair, matched)
    logits_mismatch = torch.tensor([[10.0, 0, 0, 0], [0, -10.0, -10.0, -10.0],
                                     [0, 0, 10.0, 0], [0, 0, 0, 10.0]])
    loss_mismatch = lipschitz_soft(logits_mismatch, users, entropy_table, sigma=1.0)

    logits_matched = torch.tensor([[10.0, 0, 0, 0], [-10.0, 10.0, -10.0, -10.0],
                                    [0, 0, 10.0, 0], [0, 0, 0, 10.0]])
    loss_matched = lipschitz_soft(logits_matched, users, entropy_table, sigma=1.0)
    assert loss_mismatch.item() > loss_matched.item()
    print(f"OK: lipschitz_soft penalizes a close-entropy confidence mismatch "
          f"({loss_mismatch.item():.3f}) more than a matched pair ({loss_matched.item():.3f})")


def test_floor_loss_zero_below_floor_positive_above():
    n_users = 4
    users = torch.tensor([0, 1, 2, 3])
    entropy_table = torch.linspace(0, 10, n_users)
    # uniform logits -> confidence = 1/4 = 0.25 for every TRUE class, below floor=0.3
    low_logits = torch.zeros(4, 4)
    loss_low = floor_loss(low_logits, users, entropy_table, floor=0.3)
    assert loss_low.item() < 1e-6

    # true class strongly favored -> confidence near 1, above floor=0.3
    high_logits = torch.eye(4) * 10.0
    loss_high = floor_loss(high_logits, users, entropy_table, floor=0.3)
    assert loss_high.item() > 0
    print(f"OK: floor_loss zero below floor ({loss_low.item():.2e}), positive above ({loss_high.item():.3f})")


def test_variance_of_risk_matches_manual_variance():
    logits, users, entropy_table = _toy_logits_and_users()
    conf = _per_sample_confidence(logits, users)
    expected = conf.var(unbiased=False)
    actual = variance_of_risk(logits, users, entropy_table)
    assert torch.allclose(actual, expected)
    print(f"OK: variance_of_risk ({actual.item():.4f}) matches manual population variance")


def test_group_dro_loss_equals_max_group_mean():
    n_users = 20
    users = torch.arange(n_users)
    entropy_table = torch.linspace(0, 10, n_users)
    logits = torch.eye(n_users) * 10.0  # each user perfectly "identified" to a different degree via scale
    scales = torch.linspace(1, 20, n_users).unsqueeze(1)
    logits = torch.eye(n_users) * scales
    loss = group_dro_loss(logits, users, entropy_table, n_deciles=10)

    conf = _per_sample_confidence(logits, users)
    boundaries = torch.quantile(entropy_table, torch.linspace(0, 1, 11)[1:-1])
    deciles = torch.bucketize(entropy_table, boundaries)  # entropy_table[users] == entropy_table since users=arange
    manual_max = max(conf[deciles == g].mean().item() for g in torch.unique(deciles))
    assert abs(loss.item() - manual_max) < 1e-5
    print(f"OK: group_dro_loss ({loss.item():.4f}) matches the manually computed max group mean ({manual_max:.4f})")


def test_per_user_minimax_k1_equals_max_confidence():
    logits, users, entropy_table = _toy_logits_and_users()
    conf = _per_sample_confidence(logits, users)
    loss_k1 = per_user_minimax_loss(logits, users, entropy_table, k=1)
    assert torch.allclose(loss_k1, conf.max())
    loss_k5 = per_user_minimax_loss(logits, users, entropy_table, k=5)
    assert torch.allclose(loss_k5, conf.topk(5).values.mean())
    print(f"OK: per_user_minimax_loss k=1 ({loss_k1.item():.4f}) == max confidence, "
          f"k=5 ({loss_k5.item():.4f}) == mean of top-5")


def test_decile_group_idx_matches_quantile_buckets():
    n_users = 100
    entropy_table = torch.linspace(0, 10, n_users)
    users = torch.arange(n_users)
    groups = decile_group_idx(entropy_table, users, n_groups=10)
    assert groups.min().item() == 0 and groups.max().item() == 9
    counts = torch.bincount(groups, minlength=10)
    assert (counts >= 8).all() and (counts <= 12).all()  # roughly even deciles
    print(f"OK: decile_group_idx buckets {n_users} users into 10 roughly-even groups: {counts.tolist()}")


def test_group_adversary_heads_routes_by_group():
    heads = GroupAdversaryHeads(total_users=20, n_groups=5)
    z = torch.randn(30, 256)
    group_idx = torch.randint(0, 5, (30,))
    out = heads(z, group_idx)
    assert set(out.keys()) == set(group_idx.unique().tolist())
    for g, (mask, logits) in out.items():
        assert mask.sum().item() == logits.shape[0]
        assert logits.shape[1] == 20
    print(f"OK: GroupAdversaryHeads routes {30} samples into {len(out)} present groups with correct shapes")


def test_group_adversarial_train_step_runs_and_updates_params():
    torch.manual_seed(0)
    sl, total_locations, total_users = 10, 50, 20
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=1e-3),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=1e-3),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=1e-3),
    )
    group_heads = GroupAdversaryHeads(total_users=total_users, n_groups=5)
    group_head_opt = torch.optim.Adam(group_heads.parameters(), lr=1e-3)
    entropy_table = torch.linspace(0, 10, total_users)

    batch = 32
    loc = torch.randint(1, total_locations + 1, (batch, sl))
    weekday = torch.randint(0, 7, (batch, sl))
    hour = torch.randint(0, 24, (batch, sl))
    y_next_idx = torch.randint(0, total_locations, (batch,))
    z_user_idx = torch.randint(0, total_users, (batch,))
    coord_target = torch.zeros(batch, sl, 2)
    group_idx = decile_group_idx(entropy_table, z_user_idx, n_groups=5)

    before = bundle.encoder.loc_emb.weight.clone()
    out = group_adversarial_train_step(bundle, opts, group_heads, group_head_opt,
                                        loc, weekday, hour, y_next_idx, z_user_idx, coord_target, group_idx)
    after = bundle.encoder.loc_emb.weight
    assert not torch.allclose(before, after)
    assert out["n_groups_present"] >= 1
    print(f"OK: group_adversarial_train_step ran one step, encoder params updated, "
          f"{out['n_groups_present']} groups present, worst_group_loss={out['worst_group_loss']:.3f}")


if __name__ == "__main__":
    test_lipschitz_soft_zero_when_confidence_uniform()
    test_lipschitz_soft_penalizes_similar_entropy_mismatch_more_than_dissimilar()
    test_floor_loss_zero_below_floor_positive_above()
    test_variance_of_risk_matches_manual_variance()
    test_group_dro_loss_equals_max_group_mean()
    test_per_user_minimax_k1_equals_max_confidence()
    test_decile_group_idx_matches_quantile_buckets()
    test_group_adversary_heads_routes_by_group()
    test_group_adversarial_train_step_runs_and_updates_params()
    print("All Phase 4 formulation tests passed.")
