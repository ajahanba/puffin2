"""Privacy metrics, computed from per-user risk tables -- always from a
fresh external attacker (puffin.attacks.reid), never the in-model
adversary (see puffin/attacks/README.md's threat-model table)."""
from __future__ import annotations

import numpy as np


def privacy_gain(raw_risk: np.ndarray, post_risk: np.ndarray) -> np.ndarray:
    """P(u) = 1 - R_anon(u) / R_orig(u). Where raw_risk(u) == 0 (the user was
    never identifiable to begin with), gain is defined as 0 -- there was no
    risk to reduce, not NaN/inf from a zero division."""
    raw_risk, post_risk = np.asarray(raw_risk, dtype=float), np.asarray(post_risk, dtype=float)
    ratio = np.divide(post_risk, raw_risk, out=np.zeros_like(raw_risk), where=raw_risk > 0)
    return np.where(raw_risk > 0, 1 - ratio, 0.0)


def protection_at_k(topk_acc_raw: np.ndarray, topk_acc_post: np.ndarray) -> np.ndarray:
    """"Privacy Protection@k" -- the old PUFFIN/Mo-PAE continuity metric.
    Same formula as privacy_gain, named separately because it's meant to be
    applied per-k (k=1, 5, 10) to top-k accuracy rather than once to a
    single risk number."""
    return privacy_gain(topk_acc_raw, topk_acc_post)
