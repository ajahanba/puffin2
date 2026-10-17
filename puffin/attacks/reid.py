"""Phase 2 (hardened, post-Checkpoint-2) re-identification attacker.

Two attacker families on raw, unprotected windows:
  - MLPEncoder: flattens the window (no sequence order) + MLP.
  - GRUEncoder: embedding + 2-layer GRU (same shape as
    puffin.models.mopae_torch.FixedEncoder, but GRU as the PI specified,
    not LSTM).
For each of 5 seeds, both are trained to convergence with early stopping on
an internal validation split (carved out of train, never touching test);
whichever has the higher validation accuracy for that seed is the "fresh
external attacker" for that seed (Phase 3.2.1's "strongest attacker"
principle, not an in-model adversary). Per-user test risk is averaged
across the 5 seeds' chosen attackers. Every result this module returns
carries which architecture(s) actually produced it.

`train_attacker_on_frozen_encoder` attacks the frozen OUTPUT of an
already-trained encoder (e.g. Mo-PAE-fixed) -- only an MLP head makes sense
on a single pooled feature vector, but the same early-stopping/5-seed
treatment applies.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puffin.models.mopae_torch import MLPHead


class MLPEncoder(nn.Module):
    """Flattened-window baseline: ignores sequence order entirely."""

    def __init__(self, context: int, total_locations: int, emb_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.loc_emb = nn.Embedding(total_locations + 1, emb_dim)
        self.wd_emb = nn.Embedding(8, 8)
        self.hr_emb = nn.Embedding(25, 8)
        in_dim = context * (emb_dim + 8 + 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 256), nn.ReLU(),
        )

    def forward(self, loc, weekday, hour):
        x = torch.cat([self.loc_emb(loc), self.wd_emb(weekday), self.hr_emb(hour)], dim=-1)
        return self.net(x.reshape(x.shape[0], -1))


class GRUEncoder(nn.Module):
    """Same shape as puffin.models.mopae_torch.FixedEncoder, GRU instead of LSTM."""

    def __init__(self, context: int, total_locations: int, emb_dim: int = 64):
        super().__init__()
        self.loc_emb = nn.Embedding(total_locations + 1, emb_dim)
        self.wd_emb = nn.Embedding(8, 8)
        self.hr_emb = nn.Embedding(25, 8)
        in_dim = emb_dim + 8 + 8
        self.gru1 = nn.GRU(input_size=in_dim, hidden_size=512, batch_first=True)
        self.gru2 = nn.GRU(input_size=512, hidden_size=256, batch_first=True)

    def forward(self, loc, weekday, hour):
        x = torch.cat([self.loc_emb(loc), self.wd_emb(weekday), self.hr_emb(hour)], dim=-1)
        h, _ = self.gru1(F.relu(x))
        h, _ = self.gru2(F.relu(h))
        return h[:, -1, :]


ENCODER_CLASSES = dict(mlp=MLPEncoder, gru=GRUEncoder)


def _per_user_topk(logits: torch.Tensor, user_idx: torch.Tensor, ks=(1, 5)):
    ranked = torch.argsort(logits, dim=-1, descending=True)
    rows = []
    users = user_idx.cpu().numpy()
    hit = ranked[:, :max(ks)] == user_idx.unsqueeze(1)
    for u in np.unique(users):
        mask = users == u
        out = dict(user_id=int(u) + 1, n_windows=int(mask.sum()))
        for k in ks:
            out[f"top{k}"] = float(hit[mask, :k].any(dim=1).float().mean().item())
        rows.append(out)
    return rows


def _overall_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean().item())


def _batched_forward(fn, *tensors, batch_size=4096):
    n = len(tensors[0])
    return torch.cat([fn(*(t[i:i + batch_size] for t in tensors)) for i in range(0, n, batch_size)], dim=0)


def _chronological_val_split(ts: np.ndarray, user: np.ndarray, val_frac: float, rng):
    """Per-user chronological split: for each user, the LAST val_frac of
    their windows BY TIME become validation, the rest fit -- not a random
    subsample. Checkpoint 2c (PI): early stopping on a random subsample of
    train doesn't test the thing early stopping is supposed to protect
    against here (generalizing forward in time); a user with ties in `ts`
    (e.g. coarse timestamps) breaks ties randomly via `rng`, not by array
    order, so repeated calls with different seeds don't silently pick the
    same split."""
    val_mask = np.zeros(len(ts), dtype=bool)
    tie_break = rng.random(len(ts))
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        order = idx[np.lexsort((tie_break[idx], ts[idx]))]  # sort by ts, ties broken randomly
        n_val = max(1, int(round(val_frac * len(order))))
        val_mask[order[-n_val:]] = True
    return np.where(val_mask)[0], np.where(~val_mask)[0]


def _train_sequence_attacker(encoder_cls, context: int, total_locations: int, total_users: int,
                              windows: dict, device, seed: int, max_epochs: int = 100, patience: int = 8,
                              val_frac: float = 0.1, batch_size: int = 512, lr: float = 1e-3,
                              chronological_val: bool = True):
    """Trains one encoder+head on raw sequence windows, early-stopped on an
    internal validation split carved from train. `chronological_val=True`
    (default, Checkpoint 2c): validation is each user's own chronologically
    LATEST val_frac of train windows, not a random subsample -- see
    `_chronological_val_split`. Returns (val_acc, test_per_user,
    test_overall_acc)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr = windows["train"]
    n = len(tr["cell"])
    if chronological_val and "ts" in tr and len(tr["ts"]) == n:
        val_idx, fit_idx = _chronological_val_split(tr["ts"], tr["user"], val_frac, rng)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(val_frac * n))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr, y = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"]), to_t(tr["user"])
    fit_idx_t, val_idx_t = to_t(fit_idx), to_t(val_idx)

    encoder = encoder_cls(context=context, total_locations=total_locations).to(device)
    head = MLPHead(total_users).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    best_val_acc, best_state, bad_epochs = -1.0, None, 0
    n_fit = len(fit_idx)
    for epoch in range(max_epochs):
        encoder.train()
        head.train()
        batch_perm = fit_idx_t[torch.randperm(n_fit, device=device)]
        for start in range(0, n_fit, batch_size):
            idx = batch_perm[start:start + batch_size]
            opt.zero_grad()
            logits = head(encoder(cell[idx], wd[idx], hr[idx]))
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            opt.step()

        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_logits = _batched_forward(lambda c, w, h: head(encoder(c, w, h)),
                                           cell[val_idx_t], wd[val_idx_t], hr[val_idx_t])
            val_acc = _overall_acc(val_logits, y[val_idx_t])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    encoder.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    encoder.eval()
    head.eval()

    te = windows["test"]
    te_cell, te_wd, te_hr, te_y = to_t(te["cell"]), to_t(te["weekday"]), to_t(te["hour"]), to_t(te["user"])
    with torch.no_grad():
        test_logits = _batched_forward(lambda c, w, h: head(encoder(c, w, h)), te_cell, te_wd, te_hr)
    test_overall_acc = _overall_acc(test_logits, te_y)
    test_per_user = _per_user_topk(test_logits, te_y)
    return best_val_acc, test_per_user, test_overall_acc


def train_raw_attacker_hardened(windows: dict, total_locations: int, total_users: int, device,
                                 seeds=(0, 1, 2, 3, 4), max_epochs: int = 100, patience: int = 8,
                                 chronological_val: bool = True):
    """Best-of-{MLP,GRU} per seed (chosen by validation accuracy, never by
    peeking at test), per-user test risk averaged across seeds. Returns
    (per_user_df, winner_log). per_user_df has user_id, n_windows,
    raw_top1/raw_top5 (mean across seeds), raw_top1_std, AND
    top1_seed{s}/top5_seed{s} for each seed in `seeds` -- kept (not just the
    mean) so Checkpoint 2c's seed-paired bootstrap can line up this
    function's seed-s result with train_attacker_on_frozen_encoder_hardened's
    seed-s result for the same user, rather than comparing two independent
    5-seed averages. winner_log is a list of {seed, winner, val_acc,
    test_overall_acc, mlp_val_acc, gru_val_acc}."""
    import pandas as pd
    context = windows["train"]["cell"].shape[1]
    winner_log = []
    per_seed_user_risk = []
    for seed in seeds:
        candidates = {}
        for arch, cls in ENCODER_CLASSES.items():
            val_acc, test_per_user, test_acc = _train_sequence_attacker(
                cls, context, total_locations, total_users, windows, device, seed,
                max_epochs=max_epochs, patience=patience, chronological_val=chronological_val)
            candidates[arch] = dict(val_acc=val_acc, test_per_user=test_per_user, test_acc=test_acc)
        winner = max(candidates, key=lambda a: candidates[a]["val_acc"])
        winner_log.append(dict(seed=seed, winner=winner,
                                val_acc=candidates[winner]["val_acc"],
                                test_overall_acc=candidates[winner]["test_acc"],
                                mlp_val_acc=candidates["mlp"]["val_acc"],
                                gru_val_acc=candidates["gru"]["val_acc"]))
        seed_df = pd.DataFrame(candidates[winner]["test_per_user"])[["user_id", "n_windows", "top1", "top5"]]
        seed_df = seed_df.rename(columns={"top1": f"top1_seed{seed}", "top5": f"top5_seed{seed}"})
        per_seed_user_risk.append(seed_df)

    # n_windows is deterministic given the test split (same every seed); keep
    # one copy rather than merging a column with the same name 5x.
    merged = per_seed_user_risk[0]
    for other in per_seed_user_risk[1:]:
        merged = merged.merge(other.drop(columns=["n_windows"]), on="user_id", how="outer")
    top1_cols = [c for c in merged.columns if c.startswith("top1_seed")]
    top5_cols = [c for c in merged.columns if c.startswith("top5_seed")]
    merged["raw_top1"] = merged[top1_cols].mean(axis=1)
    merged["raw_top1_std"] = merged[top1_cols].std(axis=1)
    merged["raw_top5"] = merged[top5_cols].mean(axis=1)
    out = merged[["user_id", "n_windows", "raw_top1", "raw_top1_std", "raw_top5"] + top1_cols + top5_cols]
    return out.fillna(0.0), winner_log


def train_attacker_on_frozen_encoder_hardened(encoder: nn.Module, windows: dict, total_users: int, device,
                                               seeds=(0, 1, 2, 3, 4), max_epochs: int = 100, patience: int = 8,
                                               val_frac: float = 0.1, batch_size: int = 512, lr: float = 1e-3,
                                               chronological_val: bool = True):
    """Same early-stopping/5-seed treatment as the raw attacker, but just an
    MLP head on a frozen pooled feature vector (no sequence left to feed a
    GRU). Returns per_user_df[user_id, post_top1 (mean), post_top1_std,
    top1_seed{s} for each seed] -- per-seed columns kept for Checkpoint 2c's
    seed-paired bootstrap (pair this function's seed-s result against
    train_raw_attacker_hardened's seed-s result for the same user, instead
    of comparing two independent 5-seed averages)."""
    import pandas as pd
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    tr = windows["train"]
    cell, wd, hr, y = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"]), to_t(tr["user"])
    with torch.no_grad():
        all_feat = _batched_forward(lambda c, w, h: encoder(c, w, h), cell, wd, hr)

    te = windows["test"]
    te_cell, te_wd, te_hr, te_y = to_t(te["cell"]), to_t(te["weekday"]), to_t(te["hour"]), to_t(te["user"])
    with torch.no_grad():
        te_feat = _batched_forward(lambda c, w, h: encoder(c, w, h), te_cell, te_wd, te_hr)

    per_seed_user_risk = []
    n = len(all_feat)
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if chronological_val and "ts" in tr and len(tr["ts"]) == n:
            val_idx_np, fit_idx_np = _chronological_val_split(tr["ts"], tr["user"], val_frac, rng)
        else:
            perm = rng.permutation(n)
            n_val = max(1, int(val_frac * n))
            val_idx_np, fit_idx_np = perm[:n_val], perm[n_val:]
        val_idx, fit_idx = torch.tensor(val_idx_np, device=device), torch.tensor(fit_idx_np, device=device)

        head = MLPHead(total_users).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        best_val_acc, best_state, bad_epochs = -1.0, None, 0
        n_fit = len(fit_idx)
        for epoch in range(max_epochs):
            head.train()
            batch_perm = fit_idx[torch.randperm(n_fit, device=device)]
            for start in range(0, n_fit, batch_size):
                idx = batch_perm[start:start + batch_size]
                opt.zero_grad()
                loss = F.cross_entropy(head(all_feat[idx]), y[idx])
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                val_acc = _overall_acc(head(all_feat[val_idx]), y[val_idx])
            if val_acc > best_val_acc:
                best_val_acc, best_state, bad_epochs = val_acc, copy.deepcopy(head.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            test_logits = _batched_forward(lambda f: head(f), te_feat)
        seed_df = pd.DataFrame(_per_user_topk(test_logits, te_y))[["user_id", "top1"]]
        seed_df = seed_df.rename(columns={"top1": f"top1_seed{seed}"})
        per_seed_user_risk.append(seed_df)

    merged = per_seed_user_risk[0]
    for other in per_seed_user_risk[1:]:
        merged = merged.merge(other, on="user_id", how="outer")
    top1_cols = [c for c in merged.columns if c.startswith("top1_seed")]
    merged["post_top1"] = merged[top1_cols].mean(axis=1)
    merged["post_top1_std"] = merged[top1_cols].std(axis=1)
    post_top1_cols = {c: c.replace("top1_seed", "post_top1_seed") for c in top1_cols}
    merged = merged.rename(columns=post_top1_cols)
    return merged[["user_id", "post_top1", "post_top1_std"] + list(post_top1_cols.values())].fillna(0.0)
