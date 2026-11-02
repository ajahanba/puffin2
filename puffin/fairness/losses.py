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


# --- Phase 4 formulations, evaluated under the adaptive-adversary standard
# per the PI's post-Checkpoint-3c instruction (reports/03c_uneven_gap.md).
# All share `ed_soft`'s (id_logits, z_user_idx, entropy_table) signature, so
# each is a drop-in `fairness_fn` for `fixed_train_step` -- no changes needed
# there. All use per-sample re-id confidence (softmax mass on the true user)
# as the per-sample privacy-risk proxy, same quantity `confidence_exact`
# already uses.

def _per_sample_confidence(id_logits: torch.Tensor, z_user_idx: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(id_logits, dim=-1)
    true_onehot = F.one_hot(z_user_idx, num_classes=id_logits.shape[-1]).float()
    return (true_onehot * probs).sum(-1)


def _global_decile_boundaries(entropy_table: torch.Tensor, n_deciles: int = 10) -> torch.Tensor:
    """Quantile boundaries of the full entropy table, used to bucket any
    user's entropy into one of n_deciles groups -- the same convention
    puffin.features.mobility's pd.qcut-based deciles use elsewhere, just
    computed in torch so it can run inside a training step."""
    qs = torch.linspace(0, 1, n_deciles + 1, device=entropy_table.device)[1:-1]
    return torch.quantile(entropy_table, qs)


def build_histogram_table(hist_by_user: dict, total_users: int, n_cells: int, device=None) -> torch.Tensor:
    """Location-distribution analog of build_entropy_table: hist_by_user
    keys are 1-indexed user ids (puffin.metrics.individual_fairness's
    per_user_cell_histogram convention), values are (n_cells+1,) normalized
    visitation histograms over 1-indexed cell_id (index 0 unused). Returns
    a 0-indexed (total_users, n_cells+1) tensor so it lines up with
    cross-entropy class indices (user_id - 1), same convention as
    build_entropy_table."""
    table = torch.zeros(total_users, n_cells + 1, dtype=torch.float32, device=device)
    for user_id, hist in hist_by_user.items():
        table[user_id - 1] = torch.as_tensor(hist, dtype=torch.float32, device=device)
    return table


def build_pairwise_distance_table(sim_matrix, user_ids: list, total_users: int, device=None) -> torch.Tensor:
    """Converts a (n_present_users, n_present_users) similarity matrix (as
    produced by puffin.metrics.individual_fairness's sim_* functions, over
    only the users present in that similarity computation) plus the
    user_ids list it was built from, into a full (total_users, total_users)
    0-indexed DISTANCE table (1 - similarity), so it can be indexed
    directly by z_user_idx the same way entropy_table is. Users absent from
    sim_matrix (e.g. filtered out of the train-split similarity
    computation) get distance 1.0 (maximally dissimilar) to everyone,
    including each other, as a safe default -- they should not appear in
    z_user_idx during training if the windows themselves were already
    filtered consistently, but this avoids an out-of-bounds/undefined
    value if they ever do."""
    table = torch.ones(total_users, total_users, dtype=torch.float32, device=device)
    idx = torch.as_tensor([u - 1 for u in user_ids], dtype=torch.long, device=device)
    dist = 1.0 - torch.as_tensor(sim_matrix, dtype=torch.float32, device=device)
    table[idx[:, None], idx[None, :]] = dist
    table.fill_diagonal_(0.0)
    return table


def lipschitz_soft_pairtable(id_logits: torch.Tensor, z_user_idx: torch.Tensor,
                              pairwise_distance_table: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Identical to lipschitz_soft, except the pairwise distance `d` is
    looked up from a precomputed (total_users, total_users) table instead
    of derived by broadcasting a per-user SCALAR table -- needed because a
    similarity basis like location-distribution (Jensen-Shannon distance
    over each user's visitation histogram) is not itself a per-user
    scalar, so there is no "entropy_table"-style single number to
    broadcast a pairwise difference from. The table is precomputed once
    (puffin.metrics.individual_fairness.sim_location_distribution on the
    training split, via build_pairwise_distance_table) before training
    starts, same spirit as entropy_table."""
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    d = pairwise_distance_table[z_user_idx][:, z_user_idx]
    w = torch.exp(-(d ** 2) / (2 * sigma ** 2))
    risk_diff = confidence.unsqueeze(0) - confidence.unsqueeze(1)
    return (w * risk_diff ** 2).mean()


def ed_soft_location(id_logits: torch.Tensor, z_user_idx: torch.Tensor,
                      histogram_table: torch.Tensor) -> torch.Tensor:
    """PUFFIN/ed_soft generalized from entropy to location-distribution
    similarity, to show the fairness objective is not entropy-specific:
    same structural pattern as ed_soft (expected value of the similarity
    basis under the predicted distribution should match the true user's
    own value), but the "value" is now each user's full visitation
    histogram over cells rather than a single entropy scalar, so the
    scalar MSE becomes an MSE over histogram vectors (an L2 surrogate for
    distribution closeness -- JS/KL distance itself is not a convenient
    loss term to backprop through here, unlike at evaluation time in
    puffin.metrics.individual_fairness)."""
    probs = F.softmax(id_logits, dim=-1)
    pred_hist = probs @ histogram_table  # (B, n_cells+1) expected histogram under predicted distribution
    true_hist = histogram_table[z_user_idx]  # (B, n_cells+1)
    return F.mse_loss(pred_hist, true_hist)


def lipschitz_soft(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor,
                    sigma: float = 1.0) -> torch.Tensor:
    """L_lipschitz with soft pairwise weights: an individual-fairness penalty
    (Dwork et al.-style -- similar individuals should be treated similarly)
    that pushes users with SIMILAR entropy toward SIMILAR re-id risk,
    weighted by a Gaussian kernel on entropy distance (soft) rather than
    L_pair_threshold's single global calibration target. w_ij -> 1 as the
    pair's entropy gap -> 0, -> 0 as it grows, so only genuinely similar
    users are constrained to each other."""
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    true_entropy = entropy_table[z_user_idx]
    d = true_entropy.unsqueeze(0) - true_entropy.unsqueeze(1)
    w = torch.exp(-(d ** 2) / (2 * sigma ** 2))
    risk_diff = confidence.unsqueeze(0) - confidence.unsqueeze(1)
    return (w * risk_diff ** 2).mean()


def floor_loss(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor,
               floor_multiplier: float = 1.0) -> torch.Tensor:
    """L_floor: a one-sided hinge, zero cost for any sample whose re-id
    confidence is already below the floor. Unlike a two-sided penalty, this
    cannot "level down" an already-well-protected user -- it only pushes on
    the worst-off tail, directly targeting the primary Phase 4 metric
    (worst-off-decile risk).

    The floor is `floor_multiplier / total_users` (total_users inferred
    from entropy_table's length), NOT a fixed absolute value -- an
    earlier fixed floor=0.3 default was found to be completely inert on
    GeoLife (identical outputs at lambda4 in {0.3,1.0,3.0}; confirmed via
    gradient-norm logging, scripts/diagnose_floor_loss.py, see
    CHANGES_FROM_ORIGINAL.md): the in-model adversary's confidence never
    exceeds ~0.008 there (chance is 1/180=0.0056) because the encoder's
    own adversarial training already suppresses it near chance by
    design, so relu(confidence - 0.3) was exactly zero for every sample
    throughout training. A chance-relative floor (default 1x chance
    level) gives a genuinely nonzero gradient (confirmed: roughly half
    of samples exceed it per batch) -- though its natural gradient scale
    stays small relative to the other loss terms even once active, which
    the report should state plainly rather than oversell."""
    total_users = entropy_table.shape[0]
    floor = floor_multiplier / total_users
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    return F.relu(confidence - floor).pow(2).mean()


def variance_of_risk(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor) -> torch.Tensor:
    """Variance-of-gain regularizer: penalizes dispersion of re-id confidence
    across the batch, pushing toward equalized risk regardless of its mean
    level (unlike floor_loss, this can level down a well-protected user if
    that reduces variance -- the known failure mode `leveling_down_check`
    exists to catch)."""
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    return confidence.var(unbiased=False)


def group_dro_loss(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor,
                    n_deciles: int = 10) -> torch.Tensor:
    """Group-DRO: the fairness term is the SINGLE WORST entropy-decile
    group's mean re-id confidence present in the batch (hard max over
    groups), so the encoder's gradient is driven entirely by whichever
    group is currently least protected, not the population average."""
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    true_entropy = entropy_table[z_user_idx]
    boundaries = _global_decile_boundaries(entropy_table, n_deciles)
    deciles = torch.bucketize(true_entropy, boundaries)
    group_means = []
    for g in torch.unique(deciles):
        mask = deciles == g
        group_means.append(confidence[mask].mean())
    return torch.stack(group_means).max()


def per_user_minimax_loss(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor,
                           k: int = 1) -> torch.Tensor:
    """Per-user minimax: the fairness term is the mean confidence of the
    top-k individually WORST (most identifiable) samples in the batch -- a
    finer-grained minimax than group_dro_loss's per-decile max, and the
    Phase 5 "per-user minimax" baseline the PI asked for, built on the same
    fairness_fn interface so it costs nothing extra to also screen here."""
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    k = min(k, confidence.shape[0])
    worst, _ = confidence.topk(k)
    return worst.mean()
