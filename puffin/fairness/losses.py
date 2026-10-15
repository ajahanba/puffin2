"""Fairness loss terms ported from Utkarsh Darbari's PUFFIN fork
(`models_mopae.py`, fairness="ed"/"confidence"/"matches"/"naive" branches,
on erebor under /home/NETID/ud21/Mo-PAE_og_fairness and Mo-PAE__fixed_naive).

Only `ed` and `confidence` are ported to code here, per the PI's instruction.
`matches` and `naive` are kept as reference only -- see CHANGES_FROM_ORIGINAL.md
for their original TF formulas; they are not reimplemented.

IMPORTANT FINDING (see CHANGES_FROM_ORIGINAL.md): the original `ed` loss
routes the *predicted* user's entropy through `tf.argmax` followed by
`tf.one_hot`, both of which have no gradient in TensorFlow. Every term in the
original loss4 is therefore a constant with respect to every trainable
variable -- `fairness="ed"` added a logged number but ZERO actual training
signal in both of Utkarsh's directories (that is the DEFAULT config in
`Mo-PAE_og_fairness/fairmopae.py`, i.e. the config most likely to have
produced the rejected PUFFIN draft's numbers).

`ed_exact` below reproduces that inertness faithfully (hard argmax, no
gradient) for provenance. `ed_soft` is a differentiable relaxation -- using
the id_recognizer's predicted *distribution* instead of its hard argmax -- so
that something recognizable as "the original PUFFIN loss" can actually serve
as the Phase 4 L_pair_threshold baseline, as requested. Use `ed_soft` for
real training; `ed_exact` only to document/reproduce what PUFFIN v1 did.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_entropy_table(entropy_by_user: dict, total_users: int, device=None) -> torch.Tensor:
    """entropy_by_user keys are 1-indexed user ids (as produced by
    puffin.data.geolife.compute_user_entropy); returns a 0-indexed tensor of
    length total_users so it lines up directly with cross-entropy class
    indices (user_id - 1)."""
    table = torch.zeros(total_users, dtype=torch.float32, device=device)
    for user_id, ent in entropy_by_user.items():
        table[user_id - 1] = ent
    return table


def ed_exact(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor) -> torch.Tensor:
    """Faithful port of fairness="ed". Returns a value with .requires_grad
    True (it's still part of the graph) but zero actual gradient, exactly
    like the TF original -- see module docstring. For provenance/logging
    only; do not use as a training signal."""
    with torch.no_grad():
        pred_idx = id_logits.argmax(dim=-1)
        true_entropy = entropy_table[z_user_idx]
        pred_entropy = entropy_table[pred_idx]
    return F.mse_loss(pred_entropy, true_entropy)


def ed_soft(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor) -> torch.Tensor:
    """Differentiable relaxation of `ed`: replace the hard argmax user with
    the expected user entropy under the predicted distribution, so the term
    actually provides a gradient. This is the "original PUFFIN loss" baseline
    for Phase 4 (stands in for L_pair_threshold)."""
    probs = F.softmax(id_logits, dim=-1)
    pred_entropy = probs @ entropy_table  # (B,) expected entropy under predicted distribution
    true_entropy = entropy_table[z_user_idx]
    return F.mse_loss(pred_entropy, true_entropy)


def confidence_exact(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor,
                      threshold: float = 0.1) -> torch.Tensor:
    """Faithful port of fairness="confidence". Batch-level (not per-sample):
    true_entropy/pred_entropy/confidence are each a single max over the whole
    batch, and the whole term is gated to exactly 0 whenever that batch's max
    entropy-gap already clears `threshold` -- so it is frequently inert too.
    Reference/logging only; matches the original's unusual, likely-unintended
    batch-level (rather than per-sample) reduction."""
    probs = F.softmax(id_logits, dim=-1)
    true_onehot = F.one_hot(z_user_idx, num_classes=id_logits.shape[-1]).float()

    confidence = (true_onehot * probs).sum(-1)  # predicted prob mass on the true user, per sample
    true_entropy = entropy_table[z_user_idx]  # per sample (true_onehot-weighted max == direct lookup)
    pred_entropy_products = probs * entropy_table  # (B, total_users)
    pred_entropy = pred_entropy_products.max(dim=-1).values

    entropy_diff = (true_entropy - pred_entropy) ** 2
    batch_max_diff = entropy_diff.max()
    batch_max_confidence = confidence.max()

    if batch_max_diff >= threshold:
        return torch.zeros((), device=id_logits.device)
    return batch_max_diff + batch_max_confidence
