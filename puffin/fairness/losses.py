"""Fairness loss terms for training the encoder against an in-model
re-identification adversary.

`ed_soft` is the PUFFIN entropy-difference loss: it pulls the expected
user entropy under the adversary's predicted distribution toward the
true user's own entropy, so the encoder is pushed to make low-entropy
(more identifiable) and high-entropy (less identifiable) users equally
hard to re-identify. `ed_soft_location` generalizes the same objective
from entropy to a user's full location-visitation histogram. The
remaining functions below (`lipschitz_soft`, `lipschitz_soft_pairtable`,
`floor_loss`, `variance_of_risk`, `group_dro_loss`,
`per_user_minimax_loss`) are alternative fairness objectives compared
against `ed_soft` in the paper, all sharing the same
`(id_logits, z_user_idx, entropy_table)` signature so each is a drop-in
`fairness_fn` for the training step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_entropy_table(entropy_by_user: dict, total_users: int, device=None) -> torch.Tensor:
    """entropy_by_user keys are 1-indexed user ids; returns a 0-indexed
    tensor of length total_users so it lines up directly with cross-
    entropy class indices (user_id - 1)."""
    table = torch.zeros(total_users, dtype=torch.float32, device=device)
    for user_id, ent in entropy_by_user.items():
        table[user_id - 1] = ent
    return table


def ed_soft(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor) -> torch.Tensor:
    """The PUFFIN entropy-difference loss: expected user entropy under the
    adversary's predicted distribution, pulled toward the true user's own
    entropy via MSE."""
    probs = F.softmax(id_logits, dim=-1)
    pred_entropy = probs @ entropy_table  # (B,) expected entropy under predicted distribution
    true_entropy = entropy_table[z_user_idx]
    return F.mse_loss(pred_entropy, true_entropy)


# All fairness terms below use per-sample re-id confidence (softmax mass
# on the true user) as the per-sample privacy-risk proxy.

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
    sim_matrix get distance 1.0 (maximally dissimilar) to everyone,
    including each other, as a safe default."""
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


# ed_soft_location's raw per-batch gradient norm on the encoder is ~1.67e7x
# smaller than ed_soft's (entropy vs. location-distribution similarity
# bases), because MSE over an 85,075-entry near-uniform histogram vector
# produces far smaller per-element gradients than MSE over a single
# entropy scalar at the same loss-function scale. Rescaling by this factor
# (measured via scripts/gradient_check_location_puffin.py, GeoLife, 10
# batches, batch_size=512, lambda4=1.0: ed_soft's mean encoder-gradient
# norm / ed_soft_location's = 1.666341e7) brings ed_soft_location's
# gradient magnitude in line with ed_soft's at the same lambda4, without
# changing lambda4's external interface or ed_soft_location's optimum
# (MSE's minimizer is unaffected by a positive multiplicative constant).
ED_SOFT_LOCATION_RESCALE = 1.666341e7


def ed_soft_location(id_logits: torch.Tensor, z_user_idx: torch.Tensor,
                      histogram_table: torch.Tensor, rescale: float = 1.0) -> torch.Tensor:
    """The PUFFIN entropy-difference loss generalized from entropy to
    location-distribution similarity: same structural pattern as ed_soft
    (expected value of the similarity basis under the predicted
    distribution should match the true user's own value), but the "value"
    is now each user's full visitation histogram over cells rather than a
    single entropy scalar, so the scalar MSE becomes an MSE over histogram
    vectors (an L2 surrogate for distribution closeness -- JS/KL distance
    itself is not a convenient loss term to backprop through here, unlike
    at evaluation time in puffin.metrics.individual_fairness). `rescale`
    multiplies the loss; defaults to 1.0 (unscaled) -- the gradient-
    matching factor (ED_SOFT_LOCATION_RESCALE above) must be passed
    explicitly by the caller, never silently applied by default, so every
    run's config makes clear whether it was used."""
    probs = F.softmax(id_logits, dim=-1)
    pred_hist = probs @ histogram_table  # (B, n_cells+1) expected histogram under predicted distribution
    true_hist = histogram_table[z_user_idx]  # (B, n_cells+1)
    return rescale * F.mse_loss(pred_hist, true_hist)


def lipschitz_soft(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor,
                    sigma: float = 1.0) -> torch.Tensor:
    """L_lipschitz with soft pairwise weights: an individual-fairness penalty
    (Dwork et al.-style -- similar individuals should be treated similarly)
    that pushes users with SIMILAR entropy toward SIMILAR re-id risk,
    weighted by a Gaussian kernel on entropy distance (soft) rather than a
    single global calibration target. w_ij -> 1 as the pair's entropy gap
    -> 0, -> 0 as it grows, so only genuinely similar users are constrained
    to each other."""
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
    cannot level down an already-well-protected user's risk -- it only
    pushes on the worst-off tail, directly targeting worst-off-decile risk.

    The floor is `floor_multiplier / total_users` (total_users inferred
    from entropy_table's length), NOT a fixed absolute value: a chance-
    relative floor keeps the hinge active (nonzero gradient) across
    datasets of very different population size, where a fixed absolute
    threshold could sit permanently below or above the adversary's actual
    confidence range throughout training."""
    total_users = entropy_table.shape[0]
    floor = floor_multiplier / total_users
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    return F.relu(confidence - floor).pow(2).mean()


def variance_of_risk(id_logits: torch.Tensor, z_user_idx: torch.Tensor, entropy_table: torch.Tensor) -> torch.Tensor:
    """Variance-of-gain regularizer: penalizes dispersion of re-id confidence
    across the batch, pushing toward equalized risk regardless of its mean
    level. Unlike floor_loss, this can reduce a well-protected user's
    protection if that lowers variance."""
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
    finer-grained minimax than group_dro_loss's per-decile max."""
    confidence = _per_sample_confidence(id_logits, z_user_idx)
    k = min(k, confidence.shape[0])
    worst, _ = confidence.topk(k)
    return worst.mean()
