"""The fairness-aware adaptive adversary.

The attacker in reid.py is "fairness-blind" -- it doesn't know a fairness
mechanism is active or which users it favors. This module's attacker
explicitly KNOWS the fairness objective and each user's entropy group
(decile). A mechanism that looks fair against a group-agnostic adversary
could still be unfair against one that specifically exploits the fairness
mechanism's own structure.

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
    same grouping used throughout this project's attacker evaluations."""
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


def can_infer_decile_from_features(encoder, windows: dict, feats: pd.DataFrame, device,
                                    n_deciles: int = 10, epochs: int = 20, batch_size: int = 512,
                                    lr: float = 1e-3, seed: int = 0) -> dict:
    """States the threat model for the specialized/reweighted attacker
    precisely, instead of leaving it implicit. Both attack functions above
    use `feats["s_unc"]` -- computed
    from the RAW trajectory (puffin.features.mobility) -- to assign each
    user's entropy decile. A real adversary attacking only RELEASED
    features `f` doesn't have the raw trajectory; this function checks
    whether the decile is recoverable from `f` alone, by training a small
    classifier (frozen encoder output -> 10-way decile classification) and
    comparing its accuracy to chance (1/n_deciles). High accuracy here
    means "no side information needed, the group is implicit in the
    released features themselves"; low accuracy means the specialized/
    reweighted attacker as implemented IS assuming side information (the
    true entropy label) a real adversary attacking only `f` wouldn't have.
    """
    torch.manual_seed(seed)
    decile_map = user_decile_map(feats, n_deciles)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    tr, te = windows["train"], windows["test"]
    tr_user_1idx, te_user_1idx = tr["user"] + 1, te["user"] + 1
    tr_decile = to_t(np.array([decile_map[u] for u in tr_user_1idx]))
    te_decile = np.array([decile_map[u] for u in te_user_1idx])

    cell, wd, hr = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"])
    head = _train_mlp_head_on_frozen(encoder, cell, wd, hr, tr_decile, n_deciles, device,
                                      epochs, batch_size, lr)

    te_cell, te_wd, te_hr = to_t(te["cell"]), to_t(te["weekday"]), to_t(te["hour"])
    with torch.no_grad():
        te_feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)
        pred = head(te_feat).argmax(-1).cpu().numpy()
    window_acc = float((pred == te_decile).mean())

    # Per-user majority vote -- the realistic unit for "did the adversary
    # correctly place this USER's group," not an isolated window.
    per_user_pred = pd.DataFrame({"user": te_user_1idx, "pred": pred, "true": te_decile})
    user_majority = per_user_pred.groupby("user").agg(
        pred=("pred", lambda s: s.mode().iloc[0]), true=("true", "first"))
    user_acc = float((user_majority["pred"] == user_majority["true"]).mean())

    return dict(window_level_accuracy=window_acc, user_level_majority_accuracy=user_acc,
                chance_accuracy=1.0 / n_deciles, n_deciles=n_deciles)


def _multiseed(fn, value_col: str, seeds, **kwargs) -> pd.DataFrame:
    """Shared multi-seed wrapper: runs `fn` once per seed, averages
    `value_col` (and its "top5" sibling,
    if present) per user, keeping per-seed columns too so a bootstrap over
    users downstream can use either the mean or the full per-seed spread."""
    per_seed = []
    for s in seeds:
        df = fn(seed=s, **kwargs)
        keep = ["user_id", "entropy_decile", value_col]
        top5_col = value_col.replace("top1", "top5")
        if top5_col in df.columns and top5_col != value_col:
            keep.append(top5_col)
        df = df[keep].rename(columns={value_col: f"{value_col}_seed{s}", top5_col: f"{top5_col}_seed{s}"})
        per_seed.append(df)
    merged = per_seed[0][["user_id", "entropy_decile"]].copy()
    for df in per_seed:
        merged = merged.merge(df.drop(columns=["entropy_decile"]), on="user_id", how="outer")
    top1_cols = [c for c in merged.columns if c.startswith(f"{value_col}_seed")]
    merged[value_col] = merged[top1_cols].mean(axis=1)
    merged[f"{value_col}_std"] = merged[top1_cols].std(axis=1)
    return merged.fillna(0.0)


def per_group_specialized_attack_multiseed(encoder, windows, feats, total_users, device,
                                            seeds=(0, 1, 2, 3, 4), **kwargs) -> pd.DataFrame:
    """Multi-seed wrapper around per_group_specialized_attack. Returns the
    same shape plus `specialized_top1_std` and per-seed columns."""
    return _multiseed(per_group_specialized_attack, "specialized_top1", seeds,
                       encoder=encoder, windows=windows, feats=feats, total_users=total_users,
                       device=device, **kwargs)


def group_reweighted_attack_multiseed(encoder, windows, feats, total_users, device,
                                       target_deciles, seeds=(0, 1, 2, 3, 4), **kwargs) -> pd.DataFrame:
    """Multi-seed wrapper around group_reweighted_attack. Returns the same
    shape plus `reweighted_top1_std` and per-seed columns."""
    return _multiseed(group_reweighted_attack, "reweighted_top1", seeds,
                       encoder=encoder, windows=windows, feats=feats, total_users=total_users,
                       device=device, target_deciles=target_deciles, **kwargs)
