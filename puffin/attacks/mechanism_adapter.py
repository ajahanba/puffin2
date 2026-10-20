"""Phase 5b: lets the existing naive/adaptive attack pipeline
(puffin.attacks.reid, puffin.attacks.adaptive) run against mechanisms that
have no learned encoder of their own -- GI-DP, TrajGAN -- not just Mo-PAE's.

Mo-PAE releases a FEATURE vector from its own trained encoder; the adaptive
attacker's per_group_specialized_attack/group_reweighted_attack functions
take that frozen encoder as a given. GI-DP and TrajGAN instead release
TRAJECTORIES directly (perturbed or synthetic lat/lon) -- there is no
mechanism-side encoder to hand those functions. The one new piece this
module adds is `train_encoder_on_released_windows`: an attacker-side GRU
trained end-to-end on whatever the mechanism released, exactly playing the
role Mo-PAE's own encoder plays elsewhere -- a fixed feature representation
for the SAME multi-seed specialized/reweighted attack code to run on top
of, so Checkpoint 3c's gap analysis can be computed identically across
mechanism types, apples to apples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from puffin.attacks.reid import GRUEncoder


def train_encoder_on_released_windows(windows: dict, total_locations: int, total_users: int, device,
                                       epochs: int = 25, batch_size: int = 512, lr: float = 1e-3,
                                       seed: int = 0) -> nn.Module:
    """Trains a GRUEncoder + linear classification head end-to-end to
    predict user identity directly from the released windows, then returns
    just the encoder trunk (drops the head) -- the single fixed
    representation the adaptive-attack multiseed functions then probe,
    exactly mirroring how a frozen Mo-PAE encoder is used elsewhere.
    Single-seed by design: the 5-seed rigor lives in the attack functions
    that consume this encoder's output, not in the encoder itself, same
    division of labor as a Mo-PAE run (one trained mechanism, attacked by a
    multi-seed adaptive adversary)."""
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    encoder = GRUEncoder(sl, total_locations).to(device)
    head = nn.Linear(256, total_users).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    tr = windows["train"]
    cell = torch.tensor(tr["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(tr["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(tr["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(tr["user"], dtype=torch.long, device=device)
    n = len(cell)
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = head(encoder(cell[idx], wd[idx], hr[idx]))
            loss = F.cross_entropy(logits, user_idx[idx])
            loss.backward()
            opt.step()
    encoder.eval()
    return encoder
