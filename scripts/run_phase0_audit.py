#!/usr/bin/env python
"""Phase 0 checkpoint: train legacy-mode and fixed-mode Mo-PAE ports on the
GeoLife sample and report utility / internal-re-id accuracy on train vs test,
to show how much the numbers move once issues #1-#4 and #6-#7 are fixed.

Usage:
    python scripts/run_phase0_audit.py --epochs 5 --seed 0 \
        --train-csv data/raw/geolife_sample/GeoLifeBeijing_grid_train.csv \
        --test-csv  data/raw/geolife_sample/GeoLifeBeijing_grid_test.csv \
        --out results/phase0/audit_seed0.json
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.data.geolife import load_geolife_sample, legacy_windows, fixed_windows
from puffin.models.mopae_torch import build_bundle, legacy_train_step, fixed_train_step


def topk_accuracy(logits_or_probs: torch.Tensor, target_idx: torch.Tensor, ks=(1, 5, 10)):
    ranked = torch.argsort(logits_or_probs, dim=-1, descending=True)
    out = {}
    for k in ks:
        hit = (ranked[:, :k] == target_idx.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"top{k}"] = hit
    return out


def run_legacy(df, context, epochs, seed, device, batch_size=128):
    torch.manual_seed(seed)
    total_locations = df["loc"].nunique()
    total_users = df["label"].nunique()
    (Xtr, ytr, ztr), (Xte, yte, zte) = legacy_windows(df, context, total_locations, total_users)

    bundle = build_bundle("legacy", context, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    params = (list(bundle.encoder.parameters()) + list(bundle.decoder.parameters())
              + list(bundle.id_recognizer.parameters()) + list(bundle.next_predictor.parameters()))
    optimizer = torch.optim.Adam(params, lr=3e-4)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    ztr_t = torch.tensor(ztr, dtype=torch.float32, device=device)
    n = len(Xtr_t)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            legacy_train_step(bundle, optimizer, Xtr_t[idx], ytr_t[idx], ztr_t[idx])

    def eval_split(X, y, z):
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=device)
            enc = bundle.encoder(Xt)
            next_probs = bundle.next_predictor(enc)
            id_probs = bundle.id_recognizer(enc)
            y_idx = torch.tensor(y.argmax(-1), device=device)
            z_idx = torch.tensor(z.argmax(-1), device=device)
            return dict(utility=topk_accuracy(next_probs, y_idx),
                        internal_reid=topk_accuracy(id_probs, z_idx, ks=(1, 5)))

    return dict(train=eval_split(Xtr, ytr, ztr), test=eval_split(Xte, yte, zte),
                n_train=n, n_test=len(Xte))


def run_fixed(df, context, epochs, seed, device, batch_size=128):
    torch.manual_seed(seed)
    total_locations = df["loc"].nunique()
    total_users = df["label"].nunique()
    (Xtr, ytr, ztr, wdtr, hrtr, ctr), (Xte, yte, zte, wdte, hrte, cte) = fixed_windows(
        df, context, total_locations, total_users)

    bundle = build_bundle("fixed", context, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=3e-4),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=3e-4),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=3e-4),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=3e-4),
    )

    loc_t = torch.tensor(Xtr, dtype=torch.long, device=device)
    wd_t = torch.tensor(wdtr, dtype=torch.long, device=device)
    hr_t = torch.tensor(hrtr, dtype=torch.long, device=device)
    y_idx_t = torch.tensor(ytr.argmax(-1), dtype=torch.long, device=device)
    z_idx_t = torch.tensor(ztr.argmax(-1), dtype=torch.long, device=device)
    coord_t = torch.tensor(ctr, dtype=torch.float32, device=device)
    n = len(loc_t)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            fixed_train_step(bundle, opts, loc_t[idx], wd_t[idx], hr_t[idx],
                              y_idx_t[idx], z_idx_t[idx], coord_t[idx])

    def eval_split(loc, wd, hr, y, z):
        with torch.no_grad():
            enc = bundle.encoder(torch.tensor(loc, dtype=torch.long, device=device),
                                  torch.tensor(wd, dtype=torch.long, device=device),
                                  torch.tensor(hr, dtype=torch.long, device=device))
            next_logits = bundle.next_predictor(enc)
            id_logits = bundle.id_recognizer(enc)
            y_idx = torch.tensor(y.argmax(-1), device=device)
            z_idx = torch.tensor(z.argmax(-1), device=device)
            return dict(utility=topk_accuracy(next_logits, y_idx),
                        internal_reid=topk_accuracy(id_logits, z_idx, ks=(1, 5)))

    return dict(train=eval_split(Xtr, wdtr, hrtr, ytr, ztr),
                test=eval_split(Xte, wdte, hrte, yte, zte),
                n_train=n, n_test=len(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_geolife_sample(args.train_csv, args.test_csv)
    device = torch.device(args.device)

    t0 = time.time()
    legacy_result = run_legacy(df, args.context, args.epochs, args.seed, device)
    t1 = time.time()
    fixed_result = run_fixed(df, args.context, args.epochs, args.seed, device)
    t2 = time.time()

    out = dict(
        config=vars(args),
        legacy=legacy_result,
        fixed=fixed_result,
        wall_clock_sec=dict(legacy=t1 - t0, fixed=t2 - t1),
    )
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
