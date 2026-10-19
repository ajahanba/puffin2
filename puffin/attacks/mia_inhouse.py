"""Phase 3 task 3: membership inference attack (MIA) -- in-house fallback.

POIPrivacy (https://github.com/KunlinChoi/POIPrivacy) was cloned to
legacy/POIPrivacy/ (not committed, same convention as legacy/Mo-PAE_upstream/)
and given a genuine, bounded attempt: its own README/code for all four
attacks (LocExtract, TrajExtract, LocMIA, TrajMIA) were read in full, a
fresh venv was built (legacy/POIPrivacy/.venv, this project's .venv was
never touched), its dependencies installed cleanly against our
torch==2.5.1+cu121/Python 3.10 (its pinned torch==1.12.1+cu113 wasn't even
needed), and its own `build_graph.py` demo was run end-to-end against its
bundled 4sq-NYC sample data -- it worked. So POIPrivacy is not abandoned or
broken code; it was ruled out for a structural reason instead, found by
actually reading what that demo produced:

1. **LocMIA and TrajMIA (the two attacks that map onto "is this user's data
   in the training set") both attack GETNext**, a graph-conv + transformer
   POI *recommender* (Attack3_LocMIA/model_batch.py, Attack4_TrajMIA/model.py),
   not a generic "any next-location model" interface. There is no path to
   "feed it a different model" -- the target architecture is the attack.

2. **GETNext's graph is a dense POI-to-POI adjacency matrix, written as one
   O(n_pois^2) CSV** (Attack3_LocMIA/build_graph.py's `save_graph_to_csv`:
   `np.savetxt(..., A.todense(), delimiter=',')`). Running it on the bundled
   4sq-NYC sample (our own test run) produced a 519MB CSV for ~11,000 POIs.
   GeoLife has 85,074 distinct grid cells (data/processed/geolife/stats.json)
   -- a dense 85074x85074 float32 matrix alone is ~29GB, before even running
   a GCN forward pass on it. This isn't a tuning problem; it rules out
   GeoLife outright, and GeoLife is this project's main dataset.

3. **GETNext's node features are POI categories** (`poi_catid`/`poi_catname`
   in `graph_X.csv`, one-hot encoded as a model input, plus a category-
   prediction auxiliary loss baked into the architecture) **that don't
   exist anywhere in this project's data.** `puffin/data/schema.py`'s
   unified schema (the thing every one of our processed parquets conforms
   to) has no category field at all -- meaningless for GeoLife's raw GPS
   (a GPS fix isn't a venue with a type), and even Foursquare's loader
   (`puffin/data/datasets/foursquare_unified.py`) drops whatever category
   Foursquare's raw check-ins carried, because the unified schema doesn't
   have anywhere to put it. Reconstructing it would mean fabricating label
   data this project's pipeline was deliberately designed not to need.

4. On top of both of those, LocMIA/TrajMIA are LiRA-style (likelihood-ratio
   / shadow-model) attacks -- `attack3_shadowmodel.py`/`train_shadow.py`
   expect many independently-trained copies of this same heavy
   GCN+transformer model, which multiplies problem #2's memory wall by
   however many shadow models LiRA needs.

Each of #2 and #3 is independently exactly the stop condition the plan's
time-box names ("undocumented/unrecoverable data format assumptions"), not
a dependency problem a pinned version would fix. Per the PI's explicit
time-box (PROGRESS.md), stopping here rather than rebuilding GETNext's
graph construction around a wholly different node/feature type.

## The in-house attack

Classic Yeom et al. (2018) / Shokri et al. loss-threshold MIA: a trained
model's prediction loss (equivalently, its confidence on the true label) is
systematically lower on examples it trained on than on held-out ones. This
is deliberately the SIMPLEST attack in the MIA literature, not a LiRA/
shadow-model attack like POIPrivacy's -- appropriate here because this
module's job is to demonstrate the standard member/nonmember asymmetry
exists on our data and give every later phase a real per-user signal to
build on, not to find the strongest possible attacker (that escalation, if
ever needed, is exactly where POIPrivacy's shadow-model machinery would
come back into scope).

Ground truth is `df`'s `mi_split` column, already produced by
`puffin.data.splits.membership_split`: "member" rows are trajectories the
target model is actually trained on; "nonmember" rows are per-user holdout
trajectories AND entirely held-out (`user_disjoint`) users the model never
sees at all. The attacked target model is Mo-PAE-fixed's own encoder+
next_predictor pair (`puffin.models.mopae_torch.FixedEncoder`, `MLPHead`) --
i.e. "the utility (next-location) model trained on released features f"
that `puffin/attacks/README.md`'s threat-model table names as Mo-PAE/
PUFFIN's MIA attack surface -- trained here directly on raw windows (no
adversarial/privacy loss terms: those are Mo-PAE's own defense mechanism,
`fixed_train_step`, and are orthogonal to needing *a* trained next-location
model to attack).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from puffin.attacks.windowing import build_sl_windows
from puffin.models.mopae_torch import FixedEncoder, MLPHead


def _windows_by_membership(df: pd.DataFrame, sl: int = 10, max_windows_per_user: int | None = 50,
                            seed: int = 0, respect_traj_boundary: bool = True) -> dict:
    """Builds SL-windows keyed to membership instead of the temporal
    train/test split: reuses `build_sl_windows` unchanged by relabeling
    `mi_split` ("member"/"nonmember") as its expected `split` column
    ("train"/"test") -- member rows become the target model's actual
    training windows, nonmember rows become the held-out windows the attack
    is evaluated against. Windows still never cross a (user_id, traj_id) or
    membership boundary, for the same reason `build_sl_windows` never lets
    them cross the train/test boundary: that's exactly the leakage Phase 0
    audited in upstream mopae.py (issue #2/#3)."""
    relabeled = df.copy()
    relabeled["split"] = np.where(relabeled["mi_split"] == "member", "train", "test")
    return build_sl_windows(relabeled, sl=sl, max_windows_per_user=max_windows_per_user,
                             seed=seed, respect_traj_boundary=respect_traj_boundary)


def train_target_model(windows: dict, total_locations: int, device, epochs: int = 15,
                        batch_size: int = 512, lr: float = 1e-3, seed: int = 0):
    """Trains Mo-PAE-fixed's own FixedEncoder + MLPHead directly on MEMBER
    windows only (plain cross-entropy on the true next location, no
    adversarial terms) -- this is the "target model" the attack below
    attacks. No validation/early-stopping is needed: this module's job is
    to have *a* trained next-location model whose train/held-out windows
    this attack can tell apart, not to report that model's own best
    achievable accuracy (that's Phase 2/5's job). Returns (encoder, head),
    both left in eval() mode."""
    torch.manual_seed(seed)
    sl = windows["train"]["cell"].shape[1]
    encoder = FixedEncoder(context=sl, total_locations=total_locations).to(device)
    head = MLPHead(total_locations).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    tr = windows["train"]

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr, y = to_t(tr["cell"]), to_t(tr["weekday"]), to_t(tr["hour"]), to_t(tr["next_cell"])
    n = len(cell)
    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        encoder.train()
        head.train()
        perm = torch.tensor(rng.permutation(n), device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = head(encoder(cell[idx], wd[idx], hr[idx]))
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            opt.step()

    encoder.eval()
    head.eval()
    return encoder, head


def _per_window_loss_confidence(encoder, head, window_split: dict, device, batch_size: int = 4096):
    """Per-window (not batch-averaged) cross-entropy loss and confidence
    (softmax probability assigned to the TRUE next location) -- the two
    classic Yeom/Shokri membership signals. Returns (loss, confidence,
    user) as parallel numpy arrays, one entry per window; `user` is
    windowing.py's 0-indexed array, passed through unchanged (converted
    back to the dataset's own 1-indexed user_id by the caller)."""
    n = len(window_split["cell"])
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)

    def to_t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    cell, wd, hr, y = (to_t(window_split["cell"]), to_t(window_split["weekday"]),
                       to_t(window_split["hour"]), to_t(window_split["next_cell"]))
    losses, confs = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl_ = slice(start, start + batch_size)
            logits = head(encoder(cell[sl_], wd[sl_], hr[sl_]))
            batch_loss = F.cross_entropy(logits, y[sl_], reduction="none")
            batch_conf = F.softmax(logits, dim=-1).gather(1, y[sl_].unsqueeze(1)).squeeze(1)
            losses.append(batch_loss.cpu().numpy())
            confs.append(batch_conf.cpu().numpy())
    return np.concatenate(losses), np.concatenate(confs), window_split["user"]


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The confidence threshold maximizing balanced accuracy (TPR+TNR)/2 on
    (scores, labels) -- balanced rather than plain accuracy so a skewed
    member/nonmember calibration split (likely here: `membership_split`'s
    default `nonmember_traj_frac=0.2` means member windows typically
    outnumber nonmember ones) doesn't just push the threshold to always
    predict the majority class. O(n log n) via a single sort (equivalent to
    reading the best operating point off an ROC curve), not an O(n^2) scan
    over every candidate threshold."""
    order = np.argsort(-scores)  # descending: as the cut moves down this order, more points are "predicted member"
    sorted_scores, sorted_labels = scores[order], labels[order]
    n_pos, n_neg = float(sorted_labels.sum()), float(len(sorted_labels) - sorted_labels.sum())
    if n_pos == 0 or n_neg == 0:
        return float(np.median(scores))
    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.cumsum(1 - sorted_labels)
    tpr = cum_tp / n_pos
    tnr = 1 - cum_fp / n_neg
    best_idx = int(np.argmax((tpr + tnr) / 2))
    return float(sorted_scores[best_idx])


def run_loss_threshold_mia(df: pd.DataFrame, device, sl: int = 10, max_windows_per_user: int | None = 50,
                            seed: int = 0, epochs: int = 15, batch_size: int = 512, lr: float = 1e-3,
                            calib_frac: float = 0.5, respect_traj_boundary: bool = True):
    """Phase 3 task 3's membership inference attack: trains the target model
    (`train_target_model`) on MEMBER windows, then runs a classic Yeom/
    Shokri confidence-threshold attack against it.

    The calibration split matters for not being circular: the threshold
    used to score the report below is picked on a random `calib_frac` half
    of EACH class's windows, then applied only to the other half -- so no
    window's own membership prediction was used to help set the threshold
    that produced it. (`puffin.data.splits.membership_split` already keeps
    member/nonmember windows disjoint by trajectory; this calibration split
    is a further, orthogonal split of each of those two pools.)

    Returns (per_user_df, summary):
      per_user_df: one row per user with >=1 eval window, columns
        user_id, n_member_windows, n_nonmember_windows (eval-side counts,
        i.e. after the calibration half is set aside), member_mean_confidence,
        nonmember_mean_confidence, attack_accuracy (fraction of this user's
        eval windows the calibrated threshold classifies correctly) -- a
        real per-user member/nonmember prediction, not just an aggregate
        number, per this project's whole per-user-risk methodology
        (puffin/attacks/reid.py's convention).
      summary: dict with `threshold`, `overall_auc` (confidence as score,
        membership as label, over all eval windows), `overall_accuracy`,
        `n_member_eval`, `n_nonmember_eval`.
    """
    from sklearn.metrics import roc_auc_score

    windows = _windows_by_membership(df, sl=sl, max_windows_per_user=max_windows_per_user,
                                      seed=seed, respect_traj_boundary=respect_traj_boundary)
    total_locations = int(df["cell_id"].max())
    encoder, head = train_target_model(windows, total_locations, device,
                                        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)

    _m_loss, m_conf, m_user = _per_window_loss_confidence(encoder, head, windows["train"], device)
    _n_loss, n_conf, n_user = _per_window_loss_confidence(encoder, head, windows["test"], device)

    rng = np.random.default_rng(seed)
    m_calib = rng.random(len(m_conf)) < calib_frac
    n_calib = rng.random(len(n_conf)) < calib_frac

    calib_scores = np.concatenate([m_conf[m_calib], n_conf[n_calib]])
    calib_labels = np.concatenate([np.ones(int(m_calib.sum())), np.zeros(int(n_calib.sum()))])
    threshold = _best_threshold(calib_scores, calib_labels)

    eval_m_conf, eval_m_user = m_conf[~m_calib], m_user[~m_calib]
    eval_n_conf, eval_n_user = n_conf[~n_calib], n_user[~n_calib]

    eval_scores = np.concatenate([eval_m_conf, eval_n_conf])
    eval_labels = np.concatenate([np.ones(len(eval_m_conf)), np.zeros(len(eval_n_conf))])
    overall_auc = (float(roc_auc_score(eval_labels, eval_scores))
                   if len(np.unique(eval_labels)) > 1 else float("nan"))
    overall_acc = float(((eval_scores >= threshold).astype(int) == eval_labels).mean()) if len(eval_labels) else float("nan")

    rows = []
    for u in np.unique(np.concatenate([eval_m_user, eval_n_user])):
        mm, nn = eval_m_user == u, eval_n_user == u
        n_member, n_nonmember = int(mm.sum()), int(nn.sum())
        n_correct = int((eval_m_conf[mm] >= threshold).sum()) + int((eval_n_conf[nn] < threshold).sum())
        total = n_member + n_nonmember
        rows.append(dict(
            user_id=int(u) + 1,  # windowing.py's user array is 0-indexed; the dataset's own user_id is 1-indexed
            n_member_windows=n_member,
            n_nonmember_windows=n_nonmember,
            member_mean_confidence=float(eval_m_conf[mm].mean()) if n_member else float("nan"),
            nonmember_mean_confidence=float(eval_n_conf[nn].mean()) if n_nonmember else float("nan"),
            attack_accuracy=(n_correct / total) if total else float("nan"),
        ))
    per_user_df = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)

    summary = dict(threshold=threshold, overall_auc=overall_auc, overall_accuracy=overall_acc,
                    n_member_eval=int(len(eval_m_conf)), n_nonmember_eval=int(len(eval_n_conf)))
    return per_user_df, summary
