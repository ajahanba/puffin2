"""Checkpoint 3d: decomposes the naive-vs-adaptive measurement gap.

The PI's challenge: `per_group_specialized_attack` (adaptive.py) trains
AND tests each head on only its decile's users -- a ~10x smaller
candidate pool than the naive attacker's full `total_users`-way problem
(GeoLife: ~1/18 vs ~1/180) -- and both adaptive variants assume ORACLE
knowledge of each user's true decile, which a real adversary attacking
only released features doesn't have (that's exactly what
`can_infer_decile_from_features` measures: 34-42% accuracy, not 100%).
So "specialized beats naive by 2.4x" conflates three different things:
a smaller search space, a specialized model, and an assumption the
attacker already knows what it would need to infer. This module
separates them, each as its own per-user, per-seed measurement so the
gap can be split into additive components.

Five quantities per user (each averaged over the SAME 5 attacker seeds,
so every downstream difference is a paired, not independently-resampled,
comparison -- the Checkpoint 2c pitfall this project has been careful
about since):
  (a) naive       -- the plain global attacker, full total_users-way.
  (b) restricted   -- the SAME trained naive model, logits masked to only
                       the sample's TRUE decile's user classes before
                       taking top-1 (no retraining -- isolates the
                       candidate-set-size effect alone).
  (c) specialized  -- adaptive.py's per_group_specialized_attack (oracle
                       decile routing, both train and test).
  (d) reweighted   -- adaptive.py's group_reweighted_attack, alone (not
                       maxed with (c)).
  (e) realistic    -- the specialized heads from (c), but at TEST time
                       routed by a decile CLASSIFIER's prediction (trained
                       on released features, same one
                       can_infer_decile_from_features measures), not the
                       oracle label -- what a real adversary without
                       side information actually gets.

Gap components (computed on PAIRED per-user values, bootstrapped over
users): side_information = (b)-(a), specialization = (c)-(b),
realistic = (e)-(a).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from puffin.attacks.reid import _batched_forward, _per_user_topk
from puffin.attacks.adaptive import user_decile_map, _train_mlp_head_on_frozen
from puffin.models.mopae_torch import MLPHead


def _to_t(arr, device, dtype=torch.long):
    return torch.tensor(arr, dtype=dtype, device=device)


def train_naive_with_logits(encoder, windows: dict, total_users: int, device,
                             epochs: int = 20, batch_size: int = 512, lr: float = 1e-3, seed: int = 0):
    """(a)'s model, exposing raw per-WINDOW test logits (not just the
    top1/top5 summary `_per_user_topk` reduces to) so (b) can mask them."""
    torch.manual_seed(seed)
    tr, te = windows["train"], windows["test"]
    cell, wd, hr, y = (_to_t(tr["cell"], device), _to_t(tr["weekday"], device),
                       _to_t(tr["hour"], device), _to_t(tr["user"], device))
    head = _train_mlp_head_on_frozen(encoder, cell, wd, hr, y, total_users, device, epochs, batch_size, lr)
    te_cell, te_wd, te_hr, te_y = (_to_t(te["cell"], device), _to_t(te["weekday"], device),
                                    _to_t(te["hour"], device), _to_t(te["user"], device))
    with torch.no_grad():
        te_feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
        logits = head(te_feat)
    te_user_1idx = te["user"] + 1
    return logits, te_y, te_user_1idx


def naive_per_user(logits: torch.Tensor, te_y: torch.Tensor, te_user_1idx: np.ndarray) -> pd.DataFrame:
    """(a): per-window correctness -> per-user mean (= top-1 accuracy)."""
    correct = (logits.argmax(-1) == te_y).float().cpu().numpy()
    return pd.DataFrame(dict(user_id=te_user_1idx, correct=correct)).groupby("user_id")["correct"].mean().reset_index()


def naive_restricted_per_user(logits: torch.Tensor, te_y: torch.Tensor, te_user_1idx: np.ndarray,
                               decile_map: dict) -> pd.DataFrame:
    """(b): mask each window's logits to ONLY its true decile's user
    classes, take top-1 within that restricted set. Same trained model as
    (a) -- no specialized training -- isolates the candidate-pool-size
    effect alone."""
    deciles = np.array([decile_map[u] for u in te_user_1idx])
    logits_np = logits.detach().cpu().numpy()
    te_y_np = te_y.cpu().numpy()
    pred = np.zeros(len(te_user_1idx), dtype=np.int64)
    for d in np.unique(deciles):
        group_users = np.array(sorted(u for u, dd in decile_map.items() if dd == d))
        group_idx = group_users - 1  # 0-indexed class ids matching te_y
        mask_rows = deciles == d
        sub = logits_np[mask_rows][:, group_idx]
        local_argmax = sub.argmax(axis=1)
        pred[mask_rows] = group_idx[local_argmax]
    correct = (pred == te_y_np).astype(float)
    return pd.DataFrame(dict(user_id=te_user_1idx, correct=correct)).groupby("user_id")["correct"].mean().reset_index()


def train_per_group_heads(encoder, windows: dict, feats: pd.DataFrame, total_users: int, device,
                           n_deciles: int = 10, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3,
                           seed: int = 0):
    """Same training as adaptive.per_group_specialized_attack, but returns
    the trained heads (keyed by decile) instead of a results dataframe --
    needed so (e) can reuse them with different TEST-time routing."""
    torch.manual_seed(seed)
    decile_map = user_decile_map(feats, n_deciles)
    tr = windows["train"]
    tr_user_1idx = tr["user"] + 1
    heads = {}
    for d in sorted(set(decile_map.values())):
        group_users = np.array([u for u, dd in decile_map.items() if dd == d])
        tr_mask = np.isin(tr_user_1idx, group_users)
        if tr_mask.sum() == 0:
            continue
        cell, wd, hr = (_to_t(tr["cell"][tr_mask], device), _to_t(tr["weekday"][tr_mask], device),
                         _to_t(tr["hour"][tr_mask], device))
        y = _to_t(tr["user"][tr_mask], device)
        heads[d] = _train_mlp_head_on_frozen(encoder, cell, wd, hr, y, total_users, device, epochs, batch_size, lr)
    return heads, decile_map


def specialized_per_user_oracle(encoder, windows: dict, heads: dict, decile_map: dict, device) -> pd.DataFrame:
    """(c): test-time routing by the TRUE (oracle) decile -- same result
    adaptive.per_group_specialized_attack gives, recomputed here from the
    kept heads so it's directly paired with (e) on the same trained heads."""
    te = windows["test"]
    te_user_1idx = te["user"] + 1
    deciles = np.array([decile_map[u] for u in te_user_1idx])
    te_y_np = te["user"]
    rows = []
    for d in np.unique(deciles):
        if d not in heads:
            continue
        mask = deciles == d
        te_cell, te_wd, te_hr = (_to_t(te["cell"][mask], device), _to_t(te["weekday"][mask], device),
                                  _to_t(te["hour"][mask], device))
        with torch.no_grad():
            feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
            pred = heads[d](feat).argmax(-1).cpu().numpy()
        correct = (pred == te_y_np[mask]).astype(float)
        rows.append(pd.DataFrame(dict(user_id=te_user_1idx[mask], correct=correct)))
    return pd.concat(rows).groupby("user_id")["correct"].mean().reset_index()


def train_decile_classifier(encoder, windows: dict, feats: pd.DataFrame, device,
                             n_deciles: int = 10, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3,
                             seed: int = 0):
    """Same training as adaptive.can_infer_decile_from_features, but
    returns the trained classifier head (not just its accuracy) so (e)
    can use its actual predictions to route test samples."""
    torch.manual_seed(seed)
    decile_map = user_decile_map(feats, n_deciles)
    tr = windows["train"]
    tr_user_1idx = tr["user"] + 1
    tr_decile = _to_t(np.array([decile_map[u] for u in tr_user_1idx]), device)
    cell, wd, hr = _to_t(tr["cell"], device), _to_t(tr["weekday"], device), _to_t(tr["hour"], device)
    head = _train_mlp_head_on_frozen(encoder, cell, wd, hr, tr_decile, n_deciles, device, epochs, batch_size, lr)
    return head, decile_map


def specialized_per_user_realistic(encoder, windows: dict, heads: dict, decile_classifier, device) -> pd.DataFrame:
    """(e): test-time routing by the DECILE CLASSIFIER's PREDICTION
    (trained on released features alone, per-window), not the oracle
    label -- the realistic adaptive attacker. Each window is routed
    independently (a window-level, not user-level, inference -- the
    attacker doesn't get to see a user's whole history before deciding
    which head to query on a given released window)."""
    te = windows["test"]
    te_user_1idx = te["user"] + 1
    te_y_np = te["user"]
    te_cell, te_wd, te_hr = _to_t(te["cell"], device), _to_t(te["weekday"], device), _to_t(te["hour"], device)
    with torch.no_grad():
        feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
        pred_decile = decile_classifier(feat).argmax(-1).cpu().numpy()
    rows = []
    for d in np.unique(pred_decile):
        if d not in heads:
            continue
        mask = pred_decile == d
        with torch.no_grad():
            pred_user = heads[d](feat[mask]).argmax(-1).cpu().numpy()
        correct = (pred_user == te_y_np[mask]).astype(float)
        rows.append(pd.DataFrame(dict(user_id=te_user_1idx[mask], correct=correct)))
    out = pd.concat(rows).groupby("user_id")["correct"].mean().reset_index()
    # a user whose every window got routed to a decile with no trained head
    # (can happen if the classifier predicts an empty/degenerate decile)
    # has no rows here; left for the caller to decide how to handle (we
    # report n_users actually scored, not silently zero-fill).
    return out
