"""Phase 4's new formulation (per the PI, post-Checkpoint-3c): group-aware
adversarial training. Every other Phase 4 formulation (puffin.fairness.losses)
is a regularizer added on top of ONE global in-model id_recognizer -- but
Checkpoint 3b/3c showed a per-entropy-group-SPECIALIZED external attacker
recovers risk no loss-only regularizer sees coming, because the in-model
adversary never had to face one. This formulation trains the encoder against
exactly that threat model directly: K per-entropy-decile adversary heads
instead of one global id_recognizer, with the encoder's adversarial gradient
driven by whichever group's head is CURRENTLY STRONGEST (i.e. whichever
group is currently least protected), not the population average.

This changes what `loss_priv` *is* (K heads instead of 1), not just an
additive term, so it needs its own train step rather than slotting into
`fixed_train_step`'s `fairness_fn` hook -- the encoder/decoder/next_predictor
and the overall four-sub-network, four-optimizer, autograd.grad-before-any-
step pattern (see puffin/models/mopae_torch.py's fixed_train_step docstring
for why: interleaving optimizer.step() between backward passes on a shared
retained graph corrupts saved activations) are otherwise identical.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from puffin.models.mopae_torch import MLPHead, FIXED_LAMBDAS


class GroupAdversaryHeads(torch.nn.Module):
    """K independent MLPHead re-id adversaries, one per entropy decile."""

    def __init__(self, total_users: int, n_groups: int = 10):
        super().__init__()
        self.n_groups = n_groups
        self.heads = torch.nn.ModuleList([MLPHead(total_users) for _ in range(n_groups)])

    def forward(self, z: torch.Tensor, group_idx: torch.Tensor):
        """Returns {group_id: (mask, logits)} for every group present."""
        out = {}
        for g in range(self.n_groups):
            mask = group_idx == g
            if mask.any():
                out[g] = (mask, self.heads[g](z[mask]))
        return out


def group_adversarial_train_step(bundle, opts, group_heads: GroupAdversaryHeads, group_head_opt,
                                  loc, weekday, hour, y_next_idx, z_user_idx, coord_target, group_idx,
                                  lambdas: dict = None):
    """One step: K group heads each train on their own group's samples
    (ordinary supervised re-id, maximizing their own accuracy); the encoder's
    adversarial loss uses the MIN group cross-entropy present in the batch
    (the group the adversary currently best identifies, i.e. "the strongest
    of them"), so its gradient is pushed entirely toward protecting whichever
    group is currently weakest -- mirrors fixed_train_step's
    `-l_priv * loss_priv` sign convention, just maximizing the worst group's
    loss instead of the population-average loss."""
    opts["enc"].zero_grad()
    opts["dec"].zero_grad()
    opts["next"].zero_grad()
    group_head_opt.zero_grad()

    enc = bundle.encoder(loc, weekday, hour)
    recon = bundle.decoder(enc)
    loss_recon = F.mse_loss(recon, coord_target)

    next_logits = bundle.next_predictor(enc)
    loss_util = F.cross_entropy(next_logits, y_next_idx)

    group_outputs = group_heads(enc, group_idx)
    group_losses = {g: F.cross_entropy(logits, z_user_idx[mask]) for g, (mask, logits) in group_outputs.items()}
    worst_group_loss = torch.stack(list(group_losses.values())).min()  # strongest adversary = lowest loss

    l = lambdas if lambdas is not None else FIXED_LAMBDAS
    enc_total = l["l_recon"] * loss_recon + l["l_util"] * loss_util - l["l_priv"] * worst_group_loss

    head_params = [p for g in group_losses for p in group_heads.heads[g].parameters()]
    head_grads = torch.autograd.grad(sum(group_losses.values()), head_params, retain_graph=True)
    dec_grads = torch.autograd.grad(loss_recon, list(bundle.decoder.parameters()), retain_graph=True)
    next_grads = torch.autograd.grad(loss_util, list(bundle.next_predictor.parameters()), retain_graph=True)
    enc_grads = torch.autograd.grad(enc_total, list(bundle.encoder.parameters()))

    for p, g in zip(head_params, head_grads):
        p.grad = g
    group_head_opt.step()
    for p, g in zip(bundle.decoder.parameters(), dec_grads):
        p.grad = g
    opts["dec"].step()
    for p, g in zip(bundle.next_predictor.parameters(), next_grads):
        p.grad = g
    opts["next"].step()
    for p, g in zip(bundle.encoder.parameters(), enc_grads):
        p.grad = g
    opts["enc"].step()

    return dict(loss=enc_total.item(), loss_recon=loss_recon.item(), loss_util=loss_util.item(),
                worst_group_loss=worst_group_loss.item(), n_groups_present=len(group_losses))


def decile_group_idx(entropy_table: torch.Tensor, z_user_idx: torch.Tensor, n_groups: int = 10) -> torch.Tensor:
    """Buckets each sample's true user into one of n_groups by global
    entropy-table quantiles -- same convention as puffin.fairness.losses'
    group_dro_loss, factored out here since both the train step above and
    the Phase 4 driver need it."""
    true_entropy = entropy_table[z_user_idx]
    qs = torch.linspace(0, 1, n_groups + 1, device=entropy_table.device)[1:-1]
    boundaries = torch.quantile(entropy_table, qs)
    return torch.bucketize(true_entropy, boundaries)
