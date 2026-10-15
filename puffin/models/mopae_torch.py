"""PyTorch port of Mo-PAE/PUFFIN's encoder-decoder-adversary-predictor, in two modes.

legacy mode faithfully reproduces mopae.py (see legacy/Mo-PAE_upstream/mopae.py):
  - the encoder LSTM sees the whole `context`-length window as ONE timestep
    with `context` raw location-id floats as its "features" (issue #4);
  - the id_recognizer/next_predictor emit softmax probabilities, which are
    then fed through categorical_crossentropy(..., from_logits=True) --
    i.e. softmax is applied a second time on top of already-softmaxed
    values (issue #5), reproduced here via `_double_softmax_xent`;
  - one shared Adam optimizer updates all four sub-networks (issue #7);
  - reconstruction target is the raw location-id sequence (issue #6).

fixed mode addresses the same four issues: a location (+weekday/hour)
embedding feeds a real `context`-length sequence into the encoder; next/id
heads emit logits consumed by a single, correct cross-entropy; each
sub-network gets its own optimizer; the decoder reconstructs grid (lat, lon)
coordinates instead of a location-id magnitude.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_softmax_xent(probs: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Reproduces categorical_crossentropy(y_true, y_pred, from_logits=True) when
    y_pred is already a softmax output -- the bug in upstream mopae.py (issue #5)."""
    return -(target_onehot * F.log_softmax(probs, dim=-1)).sum(-1).mean()


class LegacyEncoder(nn.Module):
    """input_shape=(n_features=1, timesteps): one timestep, `context` raw floats."""

    def __init__(self, context: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=context, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)

    def forward(self, x_loc: torch.Tensor) -> torch.Tensor:
        # x_loc: (B, context) raw location ids as floats -> (B, 1, context)
        h, _ = self.lstm1(F.relu(x_loc).unsqueeze(1))
        h, _ = self.lstm2(F.relu(h))
        return h[:, -1, :]  # (B, 256)


class FixedEncoder(nn.Module):
    """Real sequence of length `context`: embed location (+ weekday, hour) per step."""

    def __init__(self, context: int, total_locations: int, emb_dim: int = 64):
        super().__init__()
        self.loc_emb = nn.Embedding(total_locations + 1, emb_dim)
        self.wd_emb = nn.Embedding(8, 8)
        self.hr_emb = nn.Embedding(25, 8)
        in_dim = emb_dim + 8 + 8
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)

    def forward(self, loc: torch.Tensor, weekday: torch.Tensor, hour: torch.Tensor) -> torch.Tensor:
        # loc/weekday/hour: (B, context) int64
        x = torch.cat([self.loc_emb(loc), self.wd_emb(weekday), self.hr_emb(hour)], dim=-1)
        h, _ = self.lstm1(F.relu(x))
        h, _ = self.lstm2(F.relu(h))
        return h[:, -1, :]


class Decoder(nn.Module):
    """RepeatVector(context) -> LSTM(256) -> LSTM(512) -> per-step linear head."""

    def __init__(self, context: int, out_dim: int):
        super().__init__()
        self.context = context
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=512, batch_first=True)
        self.head = nn.Linear(512, out_dim)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        rep = enc_out.unsqueeze(1).repeat(1, self.context, 1)
        h, _ = self.lstm1(F.relu(rep))
        h, _ = self.lstm2(F.relu(h))
        return self.head(h)  # (B, context, out_dim)


class MLPHead(nn.Module):
    """256 -> tanh -> dropout -> 512 -> tanh -> dropout -> out_dim, logits (no final softmax)."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.Tanh(), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.Tanh(), nn.Dropout(0.3),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclasses.dataclass
class MoPAEBundle:
    encoder: nn.Module
    decoder: nn.Module
    id_recognizer: MLPHead
    next_predictor: MLPHead
    mode: str  # "legacy" | "fixed"


def build_bundle(mode: str, context: int, total_locations: int, total_users: int) -> MoPAEBundle:
    if mode == "legacy":
        encoder = LegacyEncoder(context)
        decoder = Decoder(context, out_dim=1)
    elif mode == "fixed":
        encoder = FixedEncoder(context, total_locations)
        decoder = Decoder(context, out_dim=2)  # reconstruct (lat, lon) grid coords
    else:
        raise ValueError(mode)
    return MoPAEBundle(
        encoder=encoder,
        decoder=decoder,
        id_recognizer=MLPHead(total_users),
        next_predictor=MLPHead(total_locations),
        mode=mode,
    )


# Mo-PAE sign convention, as published upstream: lambda1, lambda3 carry their own
# negative sign; loss = L1*loss1 + L2*loss2 + L3*loss3 with L1,L3 < 0 already.
LEGACY_LAMBDAS = dict(l1=-0.1, l2=0.8, l3=-0.1)
# fixed mode: explicit min-max, every lambda positive, the minus sign that makes
# this adversarial is written into the objective itself, not hidden in a sign.
FIXED_LAMBDAS = dict(l_recon=0.1, l_util=0.8, l_priv=0.1)


def legacy_train_step(bundle: MoPAEBundle, optimizer: torch.optim.Optimizer,
                       x_loc: torch.Tensor, y_next: torch.Tensor, z_user: torch.Tensor):
    """One mopae.py-faithful step: single shared optimizer, softmax heads, double-softmax xent."""
    optimizer.zero_grad()
    enc = bundle.encoder(x_loc)

    recon = bundle.decoder(enc).squeeze(-1)  # (B, context), raw linear output (matches upstream's un-activated Dense)
    raw_mse = F.mse_loss(recon, x_loc)
    log_mse = torch.log(raw_mse.clamp_min(1e-8))
    loss1 = raw_mse if log_mse < 1 else log_mse  # mopae.py's get_loss1(loss1=log_mse, loss1_1=raw_mse)

    next_probs = F.softmax(bundle.next_predictor(enc), dim=-1)
    loss2 = torch.log(_double_softmax_xent(next_probs, y_next).clamp_min(1e-8))

    id_probs = F.softmax(bundle.id_recognizer(enc), dim=-1)
    loss3 = _double_softmax_xent(id_probs, z_user)

    total = LEGACY_LAMBDAS["l1"] * loss1 + LEGACY_LAMBDAS["l2"] * loss2 + LEGACY_LAMBDAS["l3"] * loss3
    total.backward()
    optimizer.step()
    return dict(loss=total.item(), loss1=loss1.item(), loss2=loss2.item(), loss3=loss3.item())


def fixed_train_step(bundle: MoPAEBundle, opts: dict,
                      loc: torch.Tensor, weekday: torch.Tensor, hour: torch.Tensor,
                      y_next_idx: torch.Tensor, z_user_idx: torch.Tensor, coord_target: torch.Tensor,
                      fairness_fn=None, entropy_table: torch.Tensor = None, lambda4: float = 0.0):
    """Explicit min-max step: four independent optimizers, logits + single cross-entropy.

    `fairness_fn`/`entropy_table`/`lambda4` are optional and default to off, so
    Phase 0's audit numbers (run before this fairness term existed) stay
    reproducible. Pass `fairness_fn=puffin.fairness.losses.ed_soft` (the Phase
    4 L_pair_threshold baseline) and a positive `lambda4` to enable it -- it
    is added to the encoder's objective only, exactly where PUFFIN's lambda4
    lived.
    """
    for o in opts.values():
        o.zero_grad()

    enc = bundle.encoder(loc, weekday, hour)
    recon = bundle.decoder(enc)
    loss_recon = F.mse_loss(recon, coord_target)

    next_logits = bundle.next_predictor(enc)
    loss_util = F.cross_entropy(next_logits, y_next_idx)

    id_logits = bundle.id_recognizer(enc)
    loss_priv = F.cross_entropy(id_logits, z_user_idx)

    loss_fair = (fairness_fn(id_logits, z_user_idx, entropy_table)
                 if fairness_fn is not None else torch.zeros((), device=loc.device))

    # Mirrors the four independent GradientTapes in the original TF train_step:
    # compute every sub-network's gradient from the *same* pre-update forward
    # pass first (via autograd.grad, which doesn't mutate .grad or parameters),
    # then apply all four optimizer steps. Interleaving an optimizer.step()
    # between backward passes that share a retained graph corrupts the saved
    # activations those passes still need (in-place-modified-tensor error).
    id_grads = torch.autograd.grad(loss_priv, list(bundle.id_recognizer.parameters()), retain_graph=True)
    dec_grads = torch.autograd.grad(loss_recon, list(bundle.decoder.parameters()), retain_graph=True)
    next_grads = torch.autograd.grad(loss_util, list(bundle.next_predictor.parameters()), retain_graph=True)

    # encoder minimizes recon + utility but MAXIMIZES the adversary's loss (fools it);
    # the fairness term (if any) is added on top, same sign convention as PUFFIN's lambda4.
    l = FIXED_LAMBDAS
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

    return dict(loss=enc_total.item(), loss_recon=loss_recon.item(),
                loss_util=loss_util.item(), loss_priv=loss_priv.item(), loss_fair=float(loss_fair))
