"""Attacker-in-the-loop fairness: every k epochs, freeze the current
encoder and refit a FRESH attacker head -- the SAME MLPHead architecture
puffin.attacks.reid's external attacker uses, not the perpetually-weak
co-trained `id_recognizer` -- to convergence on it, then compute the
fairness loss (floor or group-DRO) from THAT head's per-user confidence
for the next k epochs of encoder training, instead of from the
co-trained adversary.

Rationale: the co-trained `id_recognizer` never gets much above chance
confidence because the encoder's own adversarial training suppresses it
the whole time, every step -- so any fairness loss computed from IT
never has a real, strong adversary to push against. A periodically-refit
head, trained to convergence on a MOMENTARILY FROZEN encoder, reaches
confidence much closer to what a real external attacker achieves on
the same features -- giving the encoder's fairness term something
real to defend against.

The co-trained id_recognizer and its own adversarial objective
(-l_priv * loss_priv) are left exactly as in fixed_train_step -- this
formulation only changes where the FAIRNESS term's signal comes from,
not Mo-PAE's main privacy objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from puffin.attacks.reid import _batched_forward
from puffin.models.mopae_torch import MLPHead, FIXED_LAMBDAS


def refit_fresh_attacker_head(encoder, cell, weekday, hour, user_idx, total_users, device,
                               epochs: int = 10, batch_size: int = 512, lr: float = 1e-3,
                               encoder_forward_batch_size: int = 4096) -> MLPHead:
    """Trains a brand-new MLPHead (not a continuation of any previous
    refit) on the CURRENT encoder's frozen output -- same role as
    puffin.attacks.reid's external attacker, just without its 5-seed/
    early-stopping treatment (this runs every k epochs inside training,
    so it needs to be fast, not exhaustively hardened).

    The encoder forward pass is chunked via `_batched_forward`
    (`encoder_forward_batch_size` rows at a time), not run on the full
    training set in one call: an unbatched call here OOM'd on Foursquare
    (~216k training windows through an LSTM encoder, a 16GB single
    allocation) despite succeeding on GeoLife's much smaller ~36k --
    this is the same encoder-forward pattern every other attacker in
    this codebase already chunks via `_batched_forward`
    (puffin/attacks/reid.py), just missed here."""
    head = MLPHead(total_users).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    with torch.no_grad():
        feat = _batched_forward(lambda c, w, h: encoder(c, w, h), cell, weekday, hour,
                                 batch_size=encoder_forward_batch_size)
    n = len(feat)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(head(feat[idx]), user_idx[idx])
            loss.backward()
            opt.step()
    head.eval()
    return head


def attacker_in_the_loop_train_step(bundle, opts, fresh_head: MLPHead, loc, weekday, hour,
                                     y_next_idx, z_user_idx, coord_target, loss_fn, entropy_table,
                                     lambda4: float, lambdas: dict = None):
    """Same four-sub-network, four-optimizer, autograd.grad-before-any-
    step structure as fixed_train_step; the only difference is the
    fairness term is computed from `fresh_head(enc)` (frozen weights,
    no optimizer of its own this step -- it only supplies a
    differentiable path back to the encoder) instead of
    `bundle.id_recognizer(enc)`."""
    for o in opts.values():
        o.zero_grad()

    enc = bundle.encoder(loc, weekday, hour)
    recon = bundle.decoder(enc)
    loss_recon = F.mse_loss(recon, coord_target)

    next_logits = bundle.next_predictor(enc)
    loss_util = F.cross_entropy(next_logits, y_next_idx)

    id_logits = bundle.id_recognizer(enc)
    loss_priv = F.cross_entropy(id_logits, z_user_idx)

    fresh_logits = fresh_head(enc)
    loss_fair = loss_fn(fresh_logits, z_user_idx, entropy_table)

    id_grads = torch.autograd.grad(loss_priv, list(bundle.id_recognizer.parameters()), retain_graph=True)
    dec_grads = torch.autograd.grad(loss_recon, list(bundle.decoder.parameters()), retain_graph=True)
    next_grads = torch.autograd.grad(loss_util, list(bundle.next_predictor.parameters()), retain_graph=True)

    l = lambdas if lambdas is not None else FIXED_LAMBDAS
    enc_total = (l["l_recon"] * loss_recon + l["l_util"] * loss_util - l["l_priv"] * loss_priv
                 + lambda4 * loss_fair)
    enc_grads = torch.autograd.grad(enc_total, list(bundle.encoder.parameters()))

    for p, g in zip(bundle.id_recognizer.parameters(), id_grads):
        p.grad = g
    opts["id"].step()
    for p, g in zip(bundle.decoder.parameters(), dec_grads):
        p.grad = g
    opts["dec"].step()
    for p, g in zip(bundle.next_predictor.parameters(), next_grads):
        p.grad = g
    opts["next"].step()
    for p, g in zip(bundle.encoder.parameters(), enc_grads):
        p.grad = g
    opts["enc"].step()

    return dict(loss_recon=loss_recon.item(), loss_util=loss_util.item(), loss_priv=loss_priv.item(),
                loss_fair=float(loss_fair))
