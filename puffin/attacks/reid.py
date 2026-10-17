"""Phase 2 task 2 / Phase 3.2.1-style re-identification attacker.

Two attacker variants, both "fresh" (trained end-to-end for this purpose,
never reusing an in-model adversary):
  - `train_raw_attacker`: attacks RAW, unprotected location windows directly
    -- this is the "no protection" baseline Phase 2 asks for, and stands in
    for Phase 3's full attack suite (which isn't built yet) using the
    strongest single attacker reasonably available at this stage: an
    embedding + 2-layer LSTM encoder, trained jointly with its own
    classification head.
  - `train_attacker_on_frozen_encoder`: attacks the OUTPUT of an already
    -trained, FROZEN encoder (e.g. a Mo-PAE-fixed encoder after Phase 2 task
    4's privacy training) -- a fresh classifier head is trained on top of
    frozen features, exactly Phase 3.2.1's "train new classifiers on f ...
    never seen during mechanism training" (the encoder itself is never
    fine-tuned here).
Both return per-user top-1/top-5 accuracy on the test split, which is what
Phase 2 calls "re-id risk."
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puffin.models.mopae_torch import FixedEncoder, MLPHead


def _per_user_topk(logits: torch.Tensor, user_idx: torch.Tensor, ks=(1, 5)):
    ranked = torch.argsort(logits, dim=-1, descending=True)
    rows = []
    users = user_idx.cpu().numpy()
    hit1 = (ranked[:, :max(ks)] == user_idx.unsqueeze(1))
    for u in np.unique(users):
        mask = users == u
        # +1: windowing.py stores 0-indexed class labels for cross_entropy;
        # report back in the dataset's native 1-indexed user_id so this
        # joins cleanly with puffin.features.mobility's output.
        out = dict(user_id=int(u) + 1, n_windows=int(mask.sum()))
        for k in ks:
            out[f"top{k}"] = float(hit1[mask, :k].any(dim=1).float().mean().item())
        rows.append(out)
    return rows


def train_raw_attacker(windows: dict, total_locations: int, total_users: int, device,
                        epochs: int = 5, batch_size: int = 512, lr: float = 1e-3):
    encoder = FixedEncoder(context=windows["train"]["cell"].shape[1], total_locations=total_locations).to(device)
    head = MLPHead(total_users).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    y = torch.tensor(tr["user"], dtype=torch.long, device=device)
    n = len(cell)

    encoder.train()
    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            feat = encoder(cell[idx], wd[idx], hr[idx])
            logits = head(feat)
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            opt.step()

    encoder.eval()
    head.eval()
    te = windows["test"]
    te_cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
    te_wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
    te_hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
    te_y = torch.tensor(te["user"], dtype=torch.long, device=device)
    with torch.no_grad():
        # Batched: an earlier version ran the full test set through the LSTM
        # encoder in one forward call, which OOM'd (188K GeoLife test
        # windows x hidden state through 2 LSTM layers tried to allocate
        # 22GB in one shot) -- see CHANGES_FROM_ORIGINAL.md.
        logits = torch.cat([head(encoder(te_cell[i:i + batch_size], te_wd[i:i + batch_size],
                                          te_hr[i:i + batch_size]))
                            for i in range(0, len(te_cell), batch_size)], dim=0)
    per_user = _per_user_topk(logits, te_y)
    return encoder, head, per_user


def train_attacker_on_frozen_encoder(encoder: nn.Module, windows: dict, total_users: int, device,
                                      epochs: int = 5, batch_size: int = 512, lr: float = 1e-3):
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    head = MLPHead(total_users).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    y = torch.tensor(tr["user"], dtype=torch.long, device=device)
    n = len(cell)

    with torch.no_grad():
        all_feat = torch.cat([encoder(cell[i:i + 4096], wd[i:i + 4096], hr[i:i + 4096])
                              for i in range(0, n, 4096)], dim=0)

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = head(all_feat[idx])
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            opt.step()

    head.eval()
    te = windows["test"]
    with torch.no_grad():
        te_cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
        te_wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
        te_hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
        te_y = torch.tensor(te["user"], dtype=torch.long, device=device)
        te_feat = torch.cat([encoder(te_cell[i:i + 4096], te_wd[i:i + 4096], te_hr[i:i + 4096])
                             for i in range(0, len(te_cell), 4096)], dim=0)
        logits = torch.cat([head(te_feat[i:i + 4096]) for i in range(0, len(te_feat), 4096)], dim=0)
    per_user = _per_user_topk(logits, te_y)
    return head, per_user
