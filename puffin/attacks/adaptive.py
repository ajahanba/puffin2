"""Phase 3 task 5 (last): the fairness-aware adaptive adversary.

Every attacker elsewhere in this project (reid.py, mia_inhouse.py,
tul_maintul.py) is "fairness-blind" -- it doesn't know a fairness mechanism
is active or which users it favors. This module's attacker explicitly
KNOWS the fairness objective and each user's entropy group (decile), per
the plan: "This answers the old paper's own limitation" -- PUFFIN v1 only
ever reported risk against a group-agnostic adversary, so a mechanism that
looks fair against that adversary could still be unfair against one that
specifically exploits the fairness mechanism's own structure.

Two variants, both attacking a FROZEN encoder (the mechanism under test):
  (a) per_group_specialized_attack: one attacker trained+evaluated PER
      entropy decile (using only that decile's own windows), to see whether
      specializing on a known group recovers MORE risk for that group than
      a single global, group-agnostic attacker does.
  (b) group_reweighted_attack: one attacker over all users, but its
      training loss is upweighted on whichever decile(s) the mechanism
      protects most (by default, the caller supplies these -- typically
      the decile(s) with the highest measured puffin.metrics.privacy.privacy_gain
      under the mechanism being tested), simulating an adversary who
      concentrates capacity exactly where protection is strongest.

Deliberately simpler than reid.py's hardened attacker (one architecture,
no early stopping, no multi-seed averaging) -- this module's job is to
test a specific KNOWN capability (group knowledge), not to re-derive the
strongest-possible-attacker machinery that already exists; a caller who
wants the full hardened treatment composes with reid.py's building blocks
instead (this module reuses GRUEncoder/MLPHead/the batched-forward helpers
from reid.py directly, not a parallel implementation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from puffin.attacks.reid import _batched_forward, _per_user_topk
from puffin.models.mopae_torch import MLPHead


def user_decile_map(feats: pd.DataFrame, n_deciles: int = 10) -> dict:
    """user_id (1-indexed) -> entropy decile (0 = lowest S_unc), from
    puffin.features.mobility.compute_mobility_features's output -- the
    same grouping Checkpoints 2/2b/2c use throughout."""
    deciles = pd.qcut(feats["s_unc"], n_deciles, labels=False, duplicates="drop")
    return dict(zip(feats["user_id"], deciles))


def _train_mlp_head_on_frozen(encoder, cell, wd, hr, y, total_users, device,
                               epochs, batch_size, lr, sample_weight=None):
    """One MLPHead on a frozen encoder's pooled output, optionally with
    per-sample loss weights (the group-reweighted variant)."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    head = MLPHead(total_users).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(cell)
    with torch.no_grad():
        feat = _batched_forward(lambda c, w, h: encoder(c, w, h), cell, wd, hr)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = head(feat[idx])
            if sample_weight is None:
                loss = F.cross_entropy(logits, y[idx])
            else:
                per_sample = F.cross_entropy(logits, y[idx], reduction="none")
                w = sample_weight[idx]
                loss = (per_sample * w).sum() / w.sum().clamp_min(1e-9)
            loss.backward()
            opt.step()
    head.eval()
    return head


def per_group_specialized_attack(encoder, windows: dict, feats: pd.DataFrame, total_users: int, device,
                                  n_deciles: int = 10, epochs: int = 20, batch_size: int = 512,
                                  lr: float = 1e-3, seed: int = 0) -> pd.DataFrame:
    """Variant (a). Returns per-user rows with `specialized_top1/top5` and
    `entropy_decile` -- compare against a global attacker's per-user risk
    for the same users (e.g. puffin.attacks.reid's output) to see whether
    knowing the group and specializing on it recovers more risk."""
    torch.manual_seed(seed)
    decile_map = user_decile_map(feats, n_deciles)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    tr, te = windows["train"], windows["test"]
    tr_user_1idx = tr["user"] + 1  # back to the dataset's own 1-indexed id, to look up decile_map
    te_user_1idx = te["user"] + 1

    rows = []
    for d in sorted(set(decile_map.values())):
        group_users = np.array([u for u, dd in decile_map.items() if dd == d])
        tr_mask = np.isin(tr_user_1idx, group_users)
        te_mask = np.isin(te_user_1idx, group_users)
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        cell, wd, hr = to_t(tr["cell"][tr_mask]), to_t(tr["weekday"][tr_mask]), to_t(tr["hour"][tr_mask])
        y = to_t(tr["user"][tr_mask])
        head = _train_mlp_head_on_frozen(encoder, cell, wd, hr, y, total_users, device, epochs, batch_size, lr)

        te_cell, te_wd, te_hr = to_t(te["cell"][te_mask]), to_t(te["weekday"][te_mask]), to_t(te["hour"][te_mask])
        te_y = to_t(te["user"][te_mask])
        with torch.no_grad():
            te_feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
            logits = head(te_feat)
        per_user = _per_user_topk(logits, te_y)
        for r in per_user:
            r["entropy_decile"] = int(d)
        rows.extend(per_user)

    out = pd.DataFrame(rows).rename(columns={"top1": "specialized_top1", "top5": "specialized_top5"})
    return out


def group_reweighted_attack(encoder, windows: dict, feats: pd.DataFrame, total_users: int, device,
                             target_deciles: list, reweight_factor: float = 5.0, n_deciles: int = 10,
                             epochs: int = 20, batch_size: int = 512, lr: float = 1e-3,
                             seed: int = 0) -> pd.DataFrame:
    """Variant (b). `target_deciles`: the group(s) the mechanism protects
    most (caller's responsibility to determine, e.g. from
    puffin.metrics.privacy.privacy_gain under the mechanism being tested).
    Returns per-user rows with `reweighted_top1/top5` and `entropy_decile`;
    compare specifically for `target_deciles`'s users against a standard
    (unweighted) attacker's risk for the same users."""
    torch.manual_seed(seed)
    decile_map = user_decile_map(feats, n_deciles)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    tr, te = windows["train"], windows["test"]
    tr_user_1idx = tr["user"] + 1
    target_users = np.array([u for u, d in decile_map.items() if d in target_deciles])
    weight = np.where(np.isin(tr_user_1idx, target_users), reweight_factor, 1.0)

    cell, wd, hr, y = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"]), to_t(tr["user"])
    sample_weight = torch.tensor(weight, dtype=torch.float32, device=device)
    head = _train_mlp_head_on_frozen(encoder, cell, wd, hr, y, total_users, device, epochs, batch_size, lr,
                                      sample_weight=sample_weight)

    te_cell, te_wd, te_hr, te_y = to_t(te["cell"]), to_t(te["weekday"]), to_t(te["hour"]), to_t(te["user"])
    with torch.no_grad():
        te_feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
        logits = head(te_feat)
    per_user = _per_user_topk(logits, te_y)
    out = pd.DataFrame(per_user).rename(columns={"top1": "reweighted_top1", "top5": "reweighted_top5"})
    out["entropy_decile"] = out["user_id"].map(decile_map)
    return out
