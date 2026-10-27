#!/usr/bin/env python
"""Diagnostic for the PI's flagged anomaly: on GeoLife, ed_soft's mean
utility (0.654) EXCEEDS the no-privacy reference's mean utility (0.623),
which should be impossible if no-privacy is a genuine utility ceiling
(it optimizes next-location prediction alone, with no competing
reconstruction/privacy/fairness objective). Trains no_privacy, plain,
and ed_soft(lambda4=1.0) side by side on GeoLife, ONE seed each, logging
held-out utility every epoch up to 2x the batch's epoch count (40, vs.
the batch's 20), to see whether no_privacy has actually converged by
epoch 20 or is still improving (undertrained ceiling) -- and whether a
longer run changes the ordering.

No early stopping exists anywhere in run_phase4d_approved_batch.py (a
fixed epoch count is used for every mechanism) -- this script's curve
is the first per-epoch visibility into that choice for this mechanism.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.fairness.losses import ed_soft, build_entropy_table
from puffin.models.mopae_torch import build_bundle, fixed_train_step

EPOCHS = 40
SEED = 0
BATCH_SIZE = 512
LR = 3e-4


def eval_utility(bundle, windows, device):
    te = windows["test"]
    cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(te["next_cell"], dtype=torch.long, device=device)
    bundle.encoder.eval()
    bundle.next_predictor.eval()
    with torch.no_grad():
        enc = bundle.encoder(cell, wd, hr)
        preds = bundle.next_predictor(enc).argmax(dim=-1)
        acc = (preds == y_next_idx).float().mean().item()
    bundle.encoder.train()
    bundle.next_predictor.train()
    return acc


def train_with_curve(mechanism, windows, total_locations, total_users, entropy_table, device,
                      lambda4=1.0, lambda_priv=0.3, seed=SEED, epochs=EPOCHS):
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=LR),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=LR),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=LR),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=LR),
    )
    if mechanism == "no_privacy":
        lambdas = dict(l_recon=0.0, l_util=1.0, l_priv=0.0)
        fairness_fn, l4 = None, 0.0
    elif mechanism == "plain":
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = None, 0.0
    elif mechanism == "ed_soft":
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = ed_soft, lambda4
    else:
        raise ValueError(mechanism)

    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(tr["next_cell"], dtype=torch.long, device=device)
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    n = len(cell)

    curve = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                              coord_target[idx], fairness_fn=fairness_fn, entropy_table=entropy_table,
                              lambda4=l4, lambdas=lambdas)
        util = eval_utility(bundle, windows, device)
        curve.append(util)
        print(f"  [{mechanism}] epoch {epoch+1}/{epochs}: utility={util:.4f}", flush=True)
    return curve


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet("data/processed/geolife/unified.parquet")
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)
    windows = build_sl_windows(df, sl=10, max_windows_per_user=200, respect_traj_boundary=True)

    curves = {}
    for mech, kwargs in [("no_privacy", {}), ("plain", {}), ("ed_soft", dict(lambda4=1.0))]:
        print(f"=== training {mech} (seed={SEED}, {EPOCHS} epochs) ===", flush=True)
        curves[mech] = train_with_curve(mech, windows, total_locations, total_users, entropy_table,
                                         device, **kwargs)

    out_dir = pathlib.Path("reports/figures/phase4e_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(curves, open(out_dir / "no_privacy_ceiling_curves.json", "w"), indent=2)

    print("\n=== summary ===")
    for mech, curve in curves.items():
        at20 = curve[19]
        best = max(curve)
        best_epoch = curve.index(best) + 1
        at_end = curve[-1]
        print(f"{mech:12s} utility@epoch20={at20:.4f}  best={best:.4f}@epoch{best_epoch}  "
              f"@epoch{EPOCHS}={at_end:.4f}  still_rising_at_20={curve[19] < curve[-1] - 0.005}")


if __name__ == "__main__":
    main()
