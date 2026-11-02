#!/usr/bin/env python
"""Unified Mo-PAE formulation driver: every fairness formulation studied
in this project trains through this one script, under one shared
protocol, so comparisons across formulations are never confounded by
different training code paths.

mechanism choices:
  plain              -- lambda4=0, any fairness_fn (inert): the
                         utility-logged plain-Mo-PAE reference.
  no_privacy         -- utility ceiling: lambda1=lambda3=lambda4=0
                         (prediction loss only, no reconstruction, no
                         privacy adversary, no fairness term).
  ed_soft            -- puffin.fairness.losses.ed_soft.
  floor              -- puffin.fairness.losses.floor_loss (the fixed,
                         chance-relative version).
  lipschitz          -- puffin.fairness.losses.lipschitz_soft.
  variance           -- puffin.fairness.losses.variance_of_risk.
  group_dro          -- puffin.fairness.losses.group_dro_loss.
  per_user_minimax   -- puffin.fairness.losses.per_user_minimax_loss.
  group_adversarial  -- puffin.fairness.group_adversarial's own train
                         step (no id_recognizer; a per-decile-group
                         adversary instead). "lambda4" here OVERRIDES
                         l_priv (the weight on -worst_group_loss),
                         since that is the knob that plays the
                         analogous role for this formulation -- noted
                         explicitly in the output so it is not read as
                         the same quantity as every other mechanism's
                         lambda4.
  attacker_in_the_loop -- puffin.fairness.attacker_in_the_loop: every
                         `--refit-every-k-epochs`, a fresh attacker head
                         is refit on the frozen encoder and used as the
                         fairness term's signal source.

Every mechanism (except no_privacy, which has no fairness term to
compare) is evaluated under the hardened external attacker, strongest
per user across MLP/GRU/GBT (puffin.attacks.reid's *_strongest_per_user
functions), with >=3 independent ENCODER-training seeds per
(dataset, mechanism, lambda4), each attacked with 5 attacker seeds.

TRAINING PROTOCOL (early stopping, chronological validation): every
mechanism trains up to `--mopae-epochs` (a ceiling, not a target) and
early-stops on a per-user chronological validation split carved from
the TRAINING windows (the LAST `--val-frac` of each user's train
windows by time, via puffin.attacks.reid's _chronological_val_split --
the same pattern already used for attacker-side early stopping, applied
here to mechanism training so every mechanism is trained to a
comparable, fairly-stopped point instead of a fixed epoch count that
can systematically favor whichever mechanism converges fastest (see
CHANGES_FROM_ORIGINAL.md for the diagnostic that found this mattering).
Patience is `--patience`, identical across every mechanism. The
held-out TEST split (windows["test"]) is never touched until final
evaluation.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.features.mobility import compute_mobility_features
from puffin.attacks.windowing import build_sl_windows
from puffin.attacks.reid import (train_raw_attacker_strongest_per_user,
                                  train_attacker_on_frozen_encoder_strongest_per_user,
                                  _batched_forward, _chronological_val_split)
from puffin.attacks.analysis import bootstrap_decile_ci
from puffin.fairness.losses import (ed_soft, floor_loss, lipschitz_soft, variance_of_risk, group_dro_loss,
                                     per_user_minimax_loss, build_entropy_table)
from puffin.fairness.attacker_in_the_loop import refit_fresh_attacker_head, attacker_in_the_loop_train_step
from puffin.fairness.group_adversarial import GroupAdversaryHeads, group_adversarial_train_step, decile_group_idx
from puffin.models.mopae_torch import build_bundle, fixed_train_step

LOSS_FORMULATIONS = dict(ed_soft=ed_soft, floor=floor_loss, lipschitz=lipschitz_soft,
                          variance=variance_of_risk, group_dro=group_dro_loss,
                          per_user_minimax=per_user_minimax_loss)


def _state_of(modules: dict) -> dict:
    """Snapshots state dicts to CPU, not a same-device deepcopy: with
    early stopping now running up to --mopae-epochs (a much higher
    ceiling than the old fixed 20), validation can improve -- and this
    get called -- dozens of times per run. A GPU-resident deepcopy on
    every improvement OOM'd in practice (confirmed: GeoLife's very first
    retraining-to-convergence job, 2026-10-30) despite the model itself
    being small, because each snapshot is a full second copy of every
    module's parameters held on the GPU alongside the live training
    copy. CPU snapshots cost ordinary RAM instead, which is not the
    scarce resource here."""
    return {name: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            for name, m in modules.items()}


def _load_state(modules: dict, state: dict):
    for name, m in modules.items():
        device = next(m.parameters()).device
        m.load_state_dict({k: v.to(device) for k, v in state[name].items()})


def _fit_val_split(tr, val_frac, seed):
    rng = np.random.default_rng(seed)
    n = len(tr["cell"])
    if "ts" in tr and len(tr["ts"]) == n and val_frac > 0:
        val_idx, fit_idx = _chronological_val_split(tr["ts"], tr["user"], val_frac, rng)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(val_frac * n)) if val_frac > 0 else 0
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
    return fit_idx, val_idx


def _batched_accuracy(encoder, next_predictor, cell, wd, hr, y_next_idx, idx_t, batch_size=1024):
    """Computes next-location top-1 accuracy WITHOUT ever concatenating
    every chunk's logits into one [n, total_locations] tensor: each
    chunk's predictions are reduced to a match count immediately, then
    discarded. GeoLife has 85,074 distinct location cells (vs.
    Foursquare's 1,668 and Gowalla's ~2,100) -- `_batched_forward`'s
    usual pattern of accumulating chunk outputs in a Python list before
    `torch.cat` OOM'd on GeoLife specifically (a 10.5GB single
    allocation) despite running fine, at the same batch size, on the
    other two datasets, because the per-row output width (total_locations)
    dominates memory here, not the row count. Smaller default batch_size
    than `_batched_forward`'s 4096 for the same reason."""
    n = len(idx_t)
    correct = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            b = idx_t[start:start + batch_size]
            logits = next_predictor(encoder(cell[b], wd[b], hr[b]))
            correct += (logits.argmax(-1) == y_next_idx[b]).sum().item()
    return correct / n


def _val_utility(encoder, next_predictor, cell, wd, hr, y_next_idx, val_idx_t, batch_size=1024):
    encoder.eval()
    next_predictor.eval()
    acc = _batched_accuracy(encoder, next_predictor, cell, wd, hr, y_next_idx, val_idx_t, batch_size)
    encoder.train()
    next_predictor.train()
    return acc


def _run_with_early_stopping(max_epochs, patience, fit_idx_t, device, batch_size,
                              train_one_epoch_fn, val_eval_fn, state_modules: dict, min_epochs=20):
    """Generic early-stopping loop shared by every mechanism: calls
    `train_one_epoch_fn(batch_idx)` once per mini-batch for one epoch
    over a fresh permutation of `fit_idx_t`, evaluates `val_eval_fn()`
    after each epoch, and keeps the best-validation-utility snapshot of
    `state_modules` (name -> nn.Module), restoring it before returning.
    Returns the number of epochs actually run.

    `min_epochs` (default 20, same for every mechanism -- not a per-
    mechanism tuning knob): plain Mo-PAE's validation utility sits flat
    at a near-chance plateau for the first ~11 epochs before breaking
    out (confirmed, scripts/diagnose_no_privacy_ceiling.py), because the
    multi-objective adversarial training takes time to find a useful
    joint direction. Without a floor, `patience` epochs of flat
    plateau is indistinguishable from genuine convergence, and early
    stopping fires during the plateau -- confirmed in practice: 5 of
    ~175 encoder-seed runs across the first full retraining-to-
    convergence batch stopped at epoch <=12 with utility stuck at the
    exact plateau value, including 2 of 10 seeds in the GeoLife
    plain-vs-ed_soft dominance comparison and 3 entire single-seed
    formulation runs (see CHANGES_FROM_ORIGINAL.md). Patience is only
    evaluated once `epoch+1 >= min_epochs`; the best-validation
    snapshot is still tracked from epoch 1, so a genuinely fast
    convergence is not penalized, only PREMATURE STOPPING during the
    plateau is prevented."""
    n_fit = len(fit_idx_t)
    best_val, best_state, bad_epochs, actual_epochs = -1.0, _state_of(state_modules), 0, 0
    for epoch in range(max_epochs):
        perm = fit_idx_t[torch.randperm(n_fit, device=device)]
        for start in range(0, n_fit, batch_size):
            train_one_epoch_fn(perm[start:start + batch_size])
        actual_epochs = epoch + 1
        val_acc = val_eval_fn()
        if val_acc > best_val:
            best_val, best_state, bad_epochs = val_acc, _state_of(state_modules), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience and actual_epochs >= min_epochs:
                break
    _load_state(state_modules, best_state)
    return actual_epochs


def train_attacker_in_the_loop_mechanism(windows, total_locations, total_users, entropy_table, lambda4,
                                          loss_variant, refit_every_k_epochs, device, max_epochs,
                                          lambda_priv=0.3, batch_size=512, lr=3e-4, seed=0,
                                          head_refit_epochs=10, head_refit_lr=1e-3,
                                          patience=8, val_frac=0.1, min_epochs=20):
    """Every `refit_every_k_epochs`, a FRESH attacker head is refit to
    convergence on the momentarily-frozen encoder (using only the FIT
    split, never validation), then used (frozen) as the fairness term's
    signal source for the next k epochs. loss_variant in
    {"floor", "group_dro"}."""
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
    loss_fn = floor_loss if loss_variant == "floor" else group_dro_loss

    tr = windows["train"]
    fit_idx, val_idx = _fit_val_split(tr, val_frac, seed)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"])
    user_idx, y_next_idx = to_t(tr["user"]), to_t(tr["next_cell"])
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    fit_idx_t, val_idx_t = to_t(fit_idx), to_t(val_idx)

    state = dict(encoder=bundle.encoder, decoder=bundle.decoder, id_recognizer=bundle.id_recognizer,
                 next_predictor=bundle.next_predictor)
    epoch_counter = {"n": 0}

    def train_batch(idx):
        if epoch_counter["n"] % refit_every_k_epochs == 0:
            train_batch.fresh_head = refit_fresh_attacker_head(
                bundle.encoder, cell[fit_idx_t], wd[fit_idx_t], hr[fit_idx_t], user_idx[fit_idx_t],
                total_users, device, epochs=head_refit_epochs, lr=head_refit_lr)
        attacker_in_the_loop_train_step(bundle, opts, train_batch.fresh_head, cell[idx], wd[idx], hr[idx],
                                         y_next_idx[idx], user_idx[idx], coord_target[idx],
                                         loss_fn, entropy_table, lambda4, lambdas)
    train_batch.fresh_head = None

    def val_eval():
        epoch_counter["n"] += 1
        return _val_utility(bundle.encoder, bundle.next_predictor, cell, wd, hr, y_next_idx, val_idx_t)

    actual_epochs = _run_with_early_stopping(max_epochs, patience, fit_idx_t, device, batch_size,
                                              train_batch, val_eval, state, min_epochs=min_epochs)
    return bundle, actual_epochs


def train_group_adversarial_mechanism(windows, total_locations, total_users, entropy_table,
                                       lambda_priv_override, device, max_epochs, n_groups=10,
                                       batch_size=512, lr=3e-4, seed=0, patience=8, val_frac=0.1,
                                       min_epochs=20):
    """No id_recognizer: a per-decile-group adversary (GroupAdversaryHeads)
    instead. lambda_priv_override plays the role every other mechanism's
    lambda_priv plays (see module docstring)."""
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.next_predictor):
        m.to(device)
    group_heads = GroupAdversaryHeads(total_users=total_users, n_groups=n_groups).to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    group_head_opt = torch.optim.Adam(group_heads.parameters(), lr=lr)
    lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv_override)

    tr = windows["train"]
    fit_idx, val_idx = _fit_val_split(tr, val_frac, seed)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"])
    user_idx, y_next_idx = to_t(tr["user"]), to_t(tr["next_cell"])
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    fit_idx_t, val_idx_t = to_t(fit_idx), to_t(val_idx)

    state = dict(encoder=bundle.encoder, decoder=bundle.decoder, next_predictor=bundle.next_predictor,
                 group_heads=group_heads)

    def train_batch(idx):
        group_idx = decile_group_idx(entropy_table, user_idx[idx], n_groups=n_groups)
        group_adversarial_train_step(bundle, opts, group_heads, group_head_opt,
                                      cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                                      coord_target[idx], group_idx, lambdas=lambdas)

    def val_eval():
        return _val_utility(bundle.encoder, bundle.next_predictor, cell, wd, hr, y_next_idx, val_idx_t)

    actual_epochs = _run_with_early_stopping(max_epochs, patience, fit_idx_t, device, batch_size,
                                              train_batch, val_eval, state, min_epochs=min_epochs)
    return bundle, actual_epochs


def train_mechanism(windows, total_locations, total_users, entropy_table, mechanism, lambda4, device,
                     max_epochs, lambda_priv=0.3, batch_size=512, lr=3e-4, seed=0,
                     loss_variant=None, refit_every_k_epochs=None, patience=8, val_frac=0.1,
                     min_epochs=20):
    if mechanism == "attacker_in_the_loop":
        return train_attacker_in_the_loop_mechanism(
            windows, total_locations, total_users, entropy_table, lambda4, loss_variant,
            refit_every_k_epochs, device, max_epochs, lambda_priv=lambda_priv, batch_size=batch_size,
            lr=lr, seed=seed, patience=patience, val_frac=val_frac, min_epochs=min_epochs)
    if mechanism == "group_adversarial":
        # lambda4 OVERRIDES l_priv for this mechanism -- see module docstring.
        return train_group_adversarial_mechanism(
            windows, total_locations, total_users, entropy_table, lambda4, device, max_epochs,
            batch_size=batch_size, lr=lr, seed=seed, patience=patience, val_frac=val_frac,
            min_epochs=min_epochs)

    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    bundle = build_bundle("fixed", sl, total_locations, total_users)
    for m in (bundle.encoder, bundle.decoder, bundle.id_recognizer, bundle.next_predictor):
        m.to(device)
    opts = dict(
        enc=torch.optim.Adam(bundle.encoder.parameters(), lr=lr),
        dec=torch.optim.Adam(bundle.decoder.parameters(), lr=lr),
        id=torch.optim.Adam(bundle.id_recognizer.parameters(), lr=lr),
        next=torch.optim.Adam(bundle.next_predictor.parameters(), lr=lr),
    )
    if mechanism == "no_privacy":
        lambdas = dict(l_recon=0.0, l_util=1.0, l_priv=0.0)
        fairness_fn, l4 = None, 0.0
    elif mechanism == "plain":
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = None, 0.0
    elif mechanism in LOSS_FORMULATIONS:
        lambdas = dict(l_recon=0.1, l_util=0.8, l_priv=lambda_priv)
        fairness_fn, l4 = LOSS_FORMULATIONS[mechanism], lambda4
    else:
        raise ValueError(mechanism)

    tr = windows["train"]
    fit_idx, val_idx = _fit_val_split(tr, val_frac, seed)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"])
    user_idx, y_next_idx = to_t(tr["user"]), to_t(tr["next_cell"])
    coord_target = torch.zeros((len(cell), sl, 2), dtype=torch.float32, device=device)
    fit_idx_t, val_idx_t = to_t(fit_idx), to_t(val_idx)

    state = dict(encoder=bundle.encoder, decoder=bundle.decoder, id_recognizer=bundle.id_recognizer,
                 next_predictor=bundle.next_predictor)

    def train_batch(idx):
        fixed_train_step(bundle, opts, cell[idx], wd[idx], hr[idx], y_next_idx[idx], user_idx[idx],
                          coord_target[idx], fairness_fn=fairness_fn, entropy_table=entropy_table,
                          lambda4=l4, lambdas=lambdas)

    def val_eval():
        return _val_utility(bundle.encoder, bundle.next_predictor, cell, wd, hr, y_next_idx, val_idx_t)

    actual_epochs = _run_with_early_stopping(max_epochs, patience, fit_idx_t, device, batch_size,
                                              train_batch, val_eval, state, min_epochs=min_epochs)
    return bundle, actual_epochs


def eval_in_model_confidence(bundle, windows, device):
    """The CO-TRAINED (in-model) adversary's own per-user confidence on
    test windows -- the quantity any floor/group-DRO-style fairness loss
    actually optimizes against, as distinct from a fresh EXTERNAL
    attacker's accuracy on the same frozen features. Not meaningful for
    group_adversarial (no id_recognizer); callers should skip it there.
    Encoder forward pass is chunked via _batched_forward -- an
    unbatched full-test-set call OOM'd on Foursquare (see
    CHANGES_FROM_ORIGINAL.md)."""
    te = windows["test"]
    cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
    user_idx = torch.tensor(te["user"], dtype=torch.long, device=device)
    bundle.encoder.eval()
    bundle.id_recognizer.eval()
    with torch.no_grad():
        id_logits = _batched_forward(lambda c, w, h: bundle.id_recognizer(bundle.encoder(c, w, h)),
                                      cell, wd, hr)
        probs = torch.softmax(id_logits, dim=-1)
        true_onehot = torch.nn.functional.one_hot(user_idx, num_classes=id_logits.shape[-1]).float()
        confidence = (true_onehot * probs).sum(-1).cpu().numpy()
    bundle.encoder.train()
    bundle.id_recognizer.train()
    user_1idx = te["user"] + 1
    return pd.DataFrame(dict(user_id=user_1idx, confidence=confidence)).groupby(
        "user_id")["confidence"].mean().reset_index()


def eval_utility(bundle, windows, device):
    """Accumulates accuracy chunk-by-chunk via _batched_accuracy, never
    concatenating the full [n_test, total_locations] logits tensor --
    see _batched_accuracy's docstring (GeoLife's 85,074-cell location
    vocabulary, ~40-50x Foursquare's and Gowalla's, made the
    concatenated-logits pattern OOM on GeoLife specifically even though
    it has far fewer windows than either other dataset)."""
    te = windows["test"]
    cell = torch.tensor(te["cell"], dtype=torch.long, device=device)
    wd = torch.tensor(te["weekday"], dtype=torch.long, device=device)
    hr = torch.tensor(te["hour"], dtype=torch.long, device=device)
    y_next_idx = torch.tensor(te["next_cell"], dtype=torch.long, device=device)
    idx_t = torch.arange(len(cell), device=device)
    bundle.encoder.eval()
    bundle.next_predictor.eval()
    acc = _batched_accuracy(bundle.encoder, bundle.next_predictor, cell, wd, hr, y_next_idx, idx_t)
    bundle.encoder.train()
    bundle.next_predictor.train()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mechanism", required=True,
                     choices=["plain", "no_privacy", "attacker_in_the_loop", "group_adversarial"]
                     + list(LOSS_FORMULATIONS.keys()))
    ap.add_argument("--loss-variant", choices=["floor", "group_dro"], default=None,
                     help="attacker_in_the_loop only")
    ap.add_argument("--refit-every-k-epochs", type=int, default=None, help="attacker_in_the_loop only")
    ap.add_argument("--sl", type=int, default=10)
    ap.add_argument("--respect-traj-boundary", action="store_true", default=True)
    ap.add_argument("--no-respect-traj-boundary", dest="respect_traj_boundary", action="store_false")
    ap.add_argument("--max-windows-per-user", type=int, default=200)
    ap.add_argument("--mopae-epochs", type=int, default=80,
                     help="MAX epochs (a ceiling); early stopping on validation utility usually ends "
                          "training sooner -- see --patience, --val-frac")
    ap.add_argument("--patience", type=int, default=8,
                     help="epochs without validation-utility improvement before stopping, same for "
                          "every mechanism")
    ap.add_argument("--min-epochs", type=int, default=20,
                     help="patience is not evaluated before this many epochs, same for every "
                          "mechanism -- plain Mo-PAE's validation utility sits flat at a near-chance "
                          "plateau for ~11 epochs before breaking out, and without this floor, "
                          "patience epochs of plateau looks identical to genuine convergence "
                          "(confirmed happening in practice; see CHANGES_FROM_ORIGINAL.md)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                     help="fraction of each user's TRAIN windows (chronologically latest) held out "
                          "for early-stopping validation; 0 disables early stopping (trains exactly "
                          "--mopae-epochs)")
    ap.add_argument("--lambda-priv", type=float, default=0.3)
    ap.add_argument("--lambda4", type=float, default=0.0,
                     help="additive fairness weight; OVERRIDES lambda-priv for group_adversarial "
                          "(see module docstring)")
    ap.add_argument("--attack-epochs", type=int, default=25)
    ap.add_argument("--encoder-seeds", type=str, default="0,1,2")
    ap.add_argument("--attacker-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    encoder_seeds = tuple(int(s) for s in args.encoder_seeds.split(","))
    attacker_seeds = tuple(int(s) for s in args.attacker_seeds.split(","))
    tag = f"{args.dataset} {args.mechanism} lambda4={args.lambda4}"
    has_id_recognizer = args.mechanism != "group_adversarial"

    df = pd.read_parquet(args.parquet)
    total_locations = int(df["cell_id"].max())
    total_users = int(df["user_id"].max())
    feats = compute_mobility_features(df)
    entropy_by_user = dict(zip(feats["user_id"], feats["s_unc"]))
    entropy_table = build_entropy_table(entropy_by_user, total_users, device=device)

    print(f"[{tag}] building windows...", flush=True)
    windows = build_sl_windows(df, sl=args.sl, max_windows_per_user=args.max_windows_per_user,
                                respect_traj_boundary=args.respect_traj_boundary)

    print(f"[{tag}] raw attacker (strongest per user, MLP+GRU+GBT, {len(attacker_seeds)} seeds)...", flush=True)
    raw_df, _ = train_raw_attacker_strongest_per_user(windows, total_locations, total_users, device,
                                                       seeds=attacker_seeds, max_epochs=args.attack_epochs,
                                                       patience=6)

    per_encoder_seed = []
    per_encoder_seed_inmodel = []
    utilities = []
    train_wall_times = []
    actual_epochs_list = []
    for enc_seed in encoder_seeds:
        print(f"[{tag}] encoder seed {enc_seed}: training mechanism (max {args.mopae_epochs} epochs, "
              f"patience {args.patience})...", flush=True)
        _t0 = time.time()
        bundle, actual_epochs = train_mechanism(
            windows, total_locations, total_users, entropy_table, args.mechanism,
            args.lambda4, device, max_epochs=args.mopae_epochs, lambda_priv=args.lambda_priv,
            seed=enc_seed, loss_variant=args.loss_variant,
            refit_every_k_epochs=args.refit_every_k_epochs, patience=args.patience, val_frac=args.val_frac,
            min_epochs=args.min_epochs)
        train_wall_times.append(time.time() - _t0)
        actual_epochs_list.append(actual_epochs)
        utility = eval_utility(bundle, windows, device)
        utilities.append(utility)
        print(f"[{tag}] encoder seed {enc_seed}: stopped after {actual_epochs} epochs, utility={utility:.4f}, "
              f"post attacker ({len(attacker_seeds)} attacker seeds)...", flush=True)
        post_df, _ = train_attacker_on_frozen_encoder_strongest_per_user(
            bundle.encoder, windows, total_users, device, seeds=attacker_seeds,
            max_epochs=args.attack_epochs, patience=6)
        post_df = post_df.rename(columns={"post_top1": f"post_top1_encseed{enc_seed}"})
        per_encoder_seed.append(post_df[["user_id", f"post_top1_encseed{enc_seed}"]])

        if has_id_recognizer:
            inmodel_df = eval_in_model_confidence(bundle, windows, device)
            inmodel_df = inmodel_df.rename(columns={"confidence": f"inmodel_confidence_encseed{enc_seed}"})
            per_encoder_seed_inmodel.append(inmodel_df[["user_id", f"inmodel_confidence_encseed{enc_seed}"]])

    merged = feats.merge(raw_df[["user_id", "n_windows", "raw_top1"]], on="user_id", how="inner")
    for df_ in per_encoder_seed:
        merged = merged.merge(df_, on="user_id", how="inner")
    for df_ in per_encoder_seed_inmodel:
        merged = merged.merge(df_, on="user_id", how="left")
    enc_cols = [c for c in merged.columns if c.startswith("post_top1_encseed")]
    merged["post_top1"] = merged[enc_cols].mean(axis=1)
    merged["post_top1_std_over_encoder_seeds"] = merged[enc_cols].std(axis=1)
    if has_id_recognizer:
        inmodel_cols = [c for c in merged.columns if c.startswith("inmodel_confidence_encseed")]
        merged["inmodel_confidence"] = merged[inmodel_cols].mean(axis=1)
    merged["entropy_decile"] = pd.qcut(merged["s_unc"], 10, labels=False, duplicates="drop")

    filtered = merged[merged["n_windows"] >= 10].copy()
    n_excluded = len(merged) - len(filtered)

    deciles = filtered["entropy_decile"].to_numpy()
    raw_ci = bootstrap_decile_ci(filtered["raw_top1"].to_numpy(), deciles)
    post_ci = bootstrap_decile_ci(filtered["post_top1"].to_numpy(), deciles)
    inmodel_ci = (bootstrap_decile_ci(filtered["inmodel_confidence"].to_numpy(), deciles)
                  if has_id_recognizer else None)
    chance_level = 1.0 / total_users

    result = dict(
        dataset=args.dataset, mechanism=args.mechanism, lambda4=args.lambda4,
        loss_variant=args.loss_variant, refit_every_k_epochs=args.refit_every_k_epochs,
        encoder_seeds=list(encoder_seeds), attacker_seeds=list(attacker_seeds),
        max_epochs=args.mopae_epochs, patience=args.patience, val_frac=args.val_frac,
        min_epochs=args.min_epochs,
        actual_epochs_per_encoder_seed=actual_epochs_list,
        utility_mean=float(np.mean(utilities)), utility_std=float(np.std(utilities)),
        utility_per_encoder_seed=utilities,
        train_wall_time_seconds_per_encoder_seed=train_wall_times,
        train_wall_time_seconds_mean=float(np.mean(train_wall_times)),
        n_users_before_filter=len(merged), n_users_after_filter=len(filtered), n_excluded=n_excluded,
        chance_level=chance_level,
        raw_decile_ci=raw_ci, post_decile_ci=post_ci, inmodel_confidence_decile_ci=inmodel_ci,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_dir / "per_user_04d.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("raw_decile_ci", "post_decile_ci", "inmodel_confidence_decile_ci")}, indent=2))


if __name__ == "__main__":
    main()
