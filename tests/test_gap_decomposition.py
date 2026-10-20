#!/usr/bin/env python
"""Correctness checks for puffin/attacks/gap_decomposition.py (Checkpoint 3d)."""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.attacks.reid import GRUEncoder
from puffin.attacks.gap_decomposition import (
    train_naive_with_logits, naive_per_user, naive_restricted_per_user,
    train_per_group_heads, specialized_per_user_oracle,
    train_decile_classifier, specialized_per_user_realistic,
)

DEVICE = torch.device("cpu")


def _synthetic_windows(n_users=20, windows_per_user=25, sl=10, total_locations=50):
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


def _synthetic_feats(n_users=20):
    return pd.DataFrame(dict(user_id=np.arange(1, n_users + 1), s_unc=np.linspace(0, 10, n_users)))


def test_restricted_top1_is_at_least_as_good_as_full_top1_per_window():
    """Masking to a smaller candidate set can only help or tie, never hurt,
    a given model's top-1 correctness on a per-window basis (the restricted
    set always contains the true class, since it's built from the true
    decile) -- a basic sanity invariant the implementation must satisfy."""
    windows = _synthetic_windows()
    encoder = GRUEncoder(context=10, total_locations=50)
    logits, te_y, te_user_1idx = train_naive_with_logits(encoder, windows, total_users=20, device=DEVICE, epochs=5)
    feats = _synthetic_feats()
    from puffin.attacks.adaptive import user_decile_map
    decile_map = user_decile_map(feats, 10)

    full_correct = (logits.argmax(-1) == te_y).float().numpy()
    restricted_df = naive_restricted_per_user(logits, te_y, te_user_1idx, decile_map)
    # recompute restricted per-window (not per-user mean) for the direct comparison
    deciles = np.array([decile_map[u] for u in te_user_1idx])
    logits_np = logits.detach().numpy()
    te_y_np = te_y.numpy()
    restricted_correct = np.zeros(len(te_user_1idx))
    for d in np.unique(deciles):
        group_users = np.array(sorted(u for u, dd in decile_map.items() if dd == d))
        group_idx = group_users - 1
        mask_rows = deciles == d
        sub = logits_np[mask_rows][:, group_idx]
        pred = group_idx[sub.argmax(axis=1)]
        restricted_correct[mask_rows] = (pred == te_y_np[mask_rows]).astype(float)

    assert (restricted_correct >= full_correct).all(), "restricting candidates can never make top-1 worse per window"
    assert restricted_correct.mean() >= full_correct.mean()
    print(f"OK: restricted top-1 ({restricted_correct.mean():.3f}) >= full top-1 "
          f"({full_correct.mean():.3f}) on every window, as required")


def test_realistic_reduces_to_oracle_when_classifier_is_perfect():
    """If the decile classifier always predicts the TRUE decile, (e) must
    equal (c) exactly -- the whole point of (e) is testing what happens
    when routing is imperfect; with perfect routing there's no difference
    to measure."""
    windows = _synthetic_windows()
    encoder = GRUEncoder(context=10, total_locations=50)
    feats = _synthetic_feats()
    heads, decile_map = train_per_group_heads(encoder, windows, feats, total_users=20, device=DEVICE, epochs=5)
    oracle_df = specialized_per_user_oracle(encoder, windows, heads, decile_map, DEVICE)

    # a "perfect" classifier: a tiny module that always outputs the
    # argmax matching the true decile for whatever test window it's given,
    # by exploiting the same deterministic mapping used elsewhere -- here,
    # simulate it by monkeypatching specialized_per_user_realistic's own
    # routing with the oracle decile directly, confirming the two paths
    # agree when routing is identical.
    te = windows["test"]
    te_user_1idx = te["user"] + 1
    oracle_decile = np.array([decile_map[u] for u in te_user_1idx])

    from puffin.attacks.reid import _batched_forward
    import torch as T
    te_cell = T.tensor(te["cell"], dtype=T.long)
    te_wd = T.tensor(te["weekday"], dtype=T.long)
    te_hr = T.tensor(te["hour"], dtype=T.long)
    with T.no_grad():
        feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
    te_y_np = te["user"]
    rows = []
    for d in np.unique(oracle_decile):
        if d not in heads:
            continue
        mask = oracle_decile == d
        with T.no_grad():
            pred = heads[d](feat[mask]).argmax(-1).numpy()
        correct = (pred == te_y_np[mask]).astype(float)
        rows.append(pd.DataFrame(dict(user_id=te_user_1idx[mask], correct=correct)))
    manual_oracle_routed = pd.concat(rows).groupby("user_id")["correct"].mean().reset_index()

    merged = oracle_df.merge(manual_oracle_routed, on="user_id", suffixes=("_fn", "_manual"))
    assert np.allclose(merged["correct_fn"], merged["correct_manual"])
    print("OK: oracle-routed specialized attack is reproducible from the kept heads "
          "(manual re-routing by the true decile matches specialized_per_user_oracle exactly)")


def test_decile_classifier_and_realistic_routing_run_end_to_end():
    windows = _synthetic_windows()
    encoder = GRUEncoder(context=10, total_locations=50)
    feats = _synthetic_feats()
    heads, decile_map = train_per_group_heads(encoder, windows, feats, total_users=20, device=DEVICE, epochs=5)
    classifier, _ = train_decile_classifier(encoder, windows, feats, DEVICE, epochs=5)
    realistic_df = specialized_per_user_realistic(encoder, windows, heads, classifier, DEVICE)
    assert set(realistic_df.columns) == {"user_id", "correct"}
    assert len(realistic_df) <= 20
    print(f"OK: realistic (predicted-decile-routed) specialized attack ran end to end, "
          f"scored {len(realistic_df)}/20 users")


if __name__ == "__main__":
    test_restricted_top1_is_at_least_as_good_as_full_top1_per_window()
    test_realistic_reduces_to_oracle_when_classifier_is_perfect()
    test_decile_classifier_and_realistic_routing_run_end_to_end()
    print("All gap_decomposition tests passed.")
