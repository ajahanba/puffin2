#!/usr/bin/env python
"""Correctness checks for puffin/baselines/trajgan.py (Phase 5, TrajGAN).

Run directly: `python tests/test_trajgan.py`.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.baselines.trajgan import (
    _trajectory_groups, _latlon_centroid_scale, _pad_batch, run_trajgan,
)
from puffin.data.schema import validate_schema


def _toy_df(n_users=10, n_traj_per_user=6, traj_len=12, seed=0):
    """Each user's trajectories sit in a small user-specific lat/lon patch
    (disjoint across users) so generated points have a real target
    distribution to be checked against, not just noise. Half of each
    user's trajectories go to split='train', half to 'test'."""
    rng = np.random.default_rng(seed)
    rows = []
    ts_counter = 0
    for u in range(1, n_users + 1):
        lat0, lon0 = 30.0 + u * 0.5, 100.0 + u * 0.5
        for traj_id in range(1, n_traj_per_user + 1):
            split = "train" if traj_id <= n_traj_per_user // 2 else "test"
            for t in range(traj_len):
                ts_counter += 1
                rows.append(dict(
                    user_id=u, traj_id=traj_id,
                    ts=pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=ts_counter),
                    lat=lat0 + rng.normal(0, 0.001), lon=lon0 + rng.normal(0, 0.001),
                    weekday=t % 7, hour=t % 24, split=split,
                ))
    return pd.DataFrame(rows)


def test_trajectory_groups_respects_split_and_truncation():
    df = _toy_df(n_users=5, n_traj_per_user=4, traj_len=20)
    train_groups = _trajectory_groups(df, "train", max_traj_len=7, min_traj_len=1)
    test_groups = _trajectory_groups(df, "test", max_traj_len=7, min_traj_len=1)
    assert len(train_groups) == 5 * 2  # 2 train trajectories/user
    assert len(test_groups) == 5 * 2
    for g in train_groups + test_groups:
        assert g["length"] == 7, "must truncate to max_traj_len"
        assert len(g["lat"]) == 7
    print("OK: _trajectory_groups splits by `split` column and truncates to max_traj_len")


def test_pad_batch_pads_at_front_and_masks_correctly():
    df = _toy_df(n_users=3, n_traj_per_user=2, traj_len=5)
    groups = _trajectory_groups(df, "train", max_traj_len=8, min_traj_len=1)
    lat_c, lon_c, scale = _latlon_centroid_scale(groups, groups)
    latlon, weekday_oh, hour_oh, mask, lengths = _pad_batch(groups, list(range(len(groups))), 8, lat_c, lon_c, scale)
    assert latlon.shape == (len(groups), 8, 2)
    assert weekday_oh.shape == (len(groups), 8, 7)
    assert hour_oh.shape == (len(groups), 8, 24)
    for b in range(len(groups)):
        L = groups[b]["length"]
        pad_len = 8 - L
        assert mask[b, :pad_len].sum() == 0, "no real signal before the pad boundary"
        assert mask[b, pad_len:].sum() == L, "every real position must be unmasked"
        assert torch.all(latlon[b, :pad_len] == 0), "padding must be exactly zero (pre-padding convention)"
        # each real timestep's one-hot weekday/hour must sum to exactly 1
        assert torch.allclose(weekday_oh[b, pad_len:].sum(dim=-1), torch.ones(L))
        assert torch.allclose(hour_oh[b, pad_len:].sum(dim=-1), torch.ones(L))
    print("OK: _pad_batch pre-pads with zeros and masks real positions correctly")


def test_end_to_end_trajgan_runs_and_learns_something():
    """The core correctness claim: training should make the generator's
    reconstruction loss (and hence lat/lon error) measurably better than
    an untrained generator on the same data -- if the embedding/fusion/
    LSTM/loss plumbing were wrong, training would not improve this at
    all (loss would be flat or NaN)."""
    torch.manual_seed(0)
    df = _toy_df(n_users=8, n_traj_per_user=6, traj_len=15, seed=1)
    device = torch.device("cpu")

    # Baseline: an untrained (0-epoch) model, to compare against.
    _, summary_untrained = run_trajgan(
        df, device, max_traj_len=15, min_traj_len=1, latent_dim=16, lstm_units=32,
        batch_size=8, max_epochs=0, lr=2e-3, seed=0)

    synthetic_df, summary = run_trajgan(
        df, device, max_traj_len=15, min_traj_len=1, latent_dim=16, lstm_units=32,
        batch_size=8, max_epochs=30, lr=2e-3, seed=0)

    for col in ("user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"):
        assert col in synthetic_df.columns, col
    validate_schema(synthetic_df)  # must be a drop-in unified-schema dataframe

    assert summary["n_train_trajectories"] == 8 * 3  # half of 6 trajs/user -> train
    assert summary["n_test_trajectories"] == 8 * 3
    assert len(summary["d_loss_history"]) == 30
    assert len(summary["g_loss_history"]) == 30
    assert np.isfinite(summary["final_d_loss"]) and np.isfinite(summary["final_g_loss"])

    # Trained generator must reconstruct real lat/lon much more closely
    # than an untrained one -- this is the real learning signal.
    assert summary["mean_abs_lat_error"] < 0.5 * summary_untrained["mean_abs_lat_error"], (
        summary["mean_abs_lat_error"], summary_untrained["mean_abs_lat_error"])
    assert summary["mean_abs_lon_error"] < 0.5 * summary_untrained["mean_abs_lon_error"]

    print(f"OK: end-to-end TrajGAN training improves lat/lon reconstruction "
          f"(untrained mean|lat err|={summary_untrained['mean_abs_lat_error']:.4f}, "
          f"trained={summary['mean_abs_lat_error']:.4f})")


if __name__ == "__main__":
    test_trajectory_groups_respects_split_and_truncation()
    test_pad_batch_pads_at_front_and_masks_correctly()
    test_end_to_end_trajgan_runs_and_learns_something()
    print("All trajgan tests passed.")
