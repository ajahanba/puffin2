"""Phase 3 task 4: trajectory-user linking (TUL) -- ported from MainTUL
(Chen et al., "Mutual Distillation Learning Network for Trajectory-User
Linking", IJCAI 2022; https://github.com/Onedean/MainTUL).

Per puffin/attacks/README.md's threat-model table, TUL attacks a mechanism
that releases actual trajectories (TrajGAN, GI-DP) -- not Mo-PAE/PUFFIN,
which releases encoder features. Neither TrajGAN nor GI-DP has been built
yet (Phase 5), so this module is validated here against GeoLife's own real
GPS trajectories, used as a stand-in "released trajectory" set -- the same
role mia_inhouse.py's windows played for MIA before a real target model
existed for that attack either.

MainTUL was cloned to legacy/MainTUL/ (not committed, same convention as
legacy/Mo-PAE_upstream/ and legacy/POIPrivacy/) and given a genuine,
bounded attempt per the plan's time-box. Unlike POIPrivacy, IT RAN: after
two trivial compatibility fixes --
  1. `utils.py`'s `EarlyStopping.__init__` used `np.Inf`, removed in
     numpy>=2.0 (`np.inf` still works) -- a pure numpy-version issue.
  2. `main.py`'s per-batch training log line referenced 5 metrics
     (acc@1/acc@5/macro_p/macro_r/macro_f1) that are only ever computed
     during the VALIDATION loop, never the training loop it's called
     from -- `output_content.format(...)` is called with 7 positional
     args against a 12-placeholder format string, raising `IndexError` on
     the very first logged batch (`batch_idx % 40 == 0` is true at
     batch_idx=0) under the authors' own default args. This is a plain
     bug, unrelated to any dependency version, so the published repo's
     own default config was apparently never actually run end-to-end in
     its current committed state; trimmed the format string to the 7
     metrics the training loop actually has.
-- its own `foursquare_mini` demo trained and tested successfully end to
end (2 epochs, batch_size 64, ~200s single-threaded CPU: test acc@1 0.573
student / 0.499 teacher, acc@5 0.684 / 0.613). See
CHANGES_FROM_ORIGINAL.md for the full account.

Because it ran, this module PORTS (does not import -- legacy/MainTUL/ is
not committed, so nothing under puffin/ may depend on it at runtime)
MainTUL's core method into this project's own schema/conventions: mutual
distillation between a lightweight LSTM "student" and a Transformer
"teacher", trained jointly (each gets its own cross-entropy loss against
the true user label, plus a bidirectional KL-divergence distillation term
between their output distributions) so the cheap student alone is good
enough to deploy as the actual attacker at inference time -- which is all
the TUL threat model needs: given ONE released trajectory, predict its
user, with no other side information.

What was ported as-is vs. simplified for this project's data, and why:

- **Category view dropped entirely.** MainTUL fuses two parallel views
  (POI id sequence, POI category sequence) via a learnable alpha-weighted
  sum in both encoders. GeoLife (this validation's dataset) has no POI
  category info at all -- a GPS fix isn't a venue with a type, exactly the
  reasoning CHANGES_FROM_ORIGINAL.md's POIPrivacy section already gave for
  why this project's unified schema (puffin/data/schema.py) has nowhere to
  put one. A constant dummy category for every point would run, but would
  only add dead parameters for zero signal while looking like real data;
  this port keeps the single POI-id view and removes the alpha-weighted
  category branch from both encoders instead, same "don't fabricate label
  data this project's pipeline doesn't have" principle.
- **Trajectory length is capped** (`max_traj_len`, default 150). GeoLife's
  real per-trip GPS trajectories run up to 92,645 points in this dataset
  (checked directly: `df.groupby(["user_id","traj_id"]).size().max()` on
  data/processed/geolife/unified.parquet) -- orders of magnitude longer
  than the check-in trajectories (tens of points/day) MainTUL was built
  for. Feeding a Transformer encoder a sequence that long is quadratic-
  attention infeasible; each trajectory is truncated to its first
  `max_traj_len` points (chronological order). A practical necessity, not
  a claim that longer context wouldn't help.
- **Train/test split reuses this project's own `mi_split` column**
  (puffin.data.splits.membership_split: whole trajectories held out per
  user, never split mid-sequence) instead of re-deriving MainTUL's own ad
  hoc 80/20 `split_dataset` -- same number by default
  (`nonmember_traj_frac=0.2`) and the same "whole trajectory" property,
  reusing existing infra instead of duplicating it (the same pattern
  `mia_inhouse.py`'s `_windows_by_membership` already established).
  `user_disjoint` users (entirely held-out, membership_split's user-level
  non-membership) are EXCLUDED here, unlike mia_inhouse.py: TUL is a
  CLOSED-WORLD classification problem (the classifier's output layer has
  one slot per training user), so a disjoint user's trajectory is
  guaranteed-wrong by construction, not a real attack signal -- MIA's
  open-world threshold test has no such constraint, which is why
  mia_inhouse.py keeps them.
- **Long-term augmentation kept** -- MainTUL's actual "mutual
  distillation" trick: during TRAINING ONLY, the teacher sees the
  student's trajectory concatenated with up to `long_term_num` other
  random trajectories of the SAME user (sampled from that user's own full
  train pool, exactly as the original's `long_term_augment` samples from
  `user_traj_train[user_id]` regardless of the train/valid subsampler
  split -- this port's validation split has the same property, not a new
  deviation), giving the teacher richer context to produce better soft
  targets for the student to distill from. At TEST time neither network
  uses any side information: both run on the single trajectory being
  attacked, alone. The original crashes (`random.sample` on a population
  smaller than `long_term_num`) for any user with too few training
  trajectories; here `long_term_num` is clamped to
  `min(long_term_num, n_train_trajs - 1)` per user instead (46/181 GeoLife
  users have fewer than 9 train trajectories after the 80/20-equivalent
  split).
- **Temporal encoding uses position index, not a raw timestamp.**
  MainTUL's `TemporalEncoding` takes `cos(timestamp * w + b)` where
  `timestamp` is literally the raw check-in unix time. This project's `ts`
  is a `datetime64`, not comparable seconds-since-epoch across users in a
  way that would matter here, and the only real use the original makes of
  a large arbitrary number is to pass it through a learnable per-dimension
  frequency `w`; a 0-indexed WITHIN-SEQUENCE position does the same job
  (relative order along the sequence) without inventing a cross-user
  absolute-time meaning the architecture doesn't otherwise rely on.
- **Not ported**: `--times` multi-seed repeats, file logging, loss-curve
  plotting, multi-GPU `DataParallel`, MainTUL's own file-based checkpoint
  (`torch.save` to disk mid-training) -- all experiment-harness
  scaffolding, not the method; this module uses the in-memory best-state
  pattern (`copy.deepcopy` + early stopping on validation accuracy)
  `puffin.attacks.reid` already uses. Also switched `F.kl_div`'s
  deprecated `size_average=False` to `reduction="batchmean"` (same
  computation, no deprecation warning).

This IS structurally different from `puffin.attacks.reid`'s re-id
attacker, not just the same thing relabeled: `reid.py`'s GRU/MLP attackers
always consume a FIXED-length SL window (sliding windows carved out of a
trajectory, trajectory identity discarded) -- never padded, never masked,
because every input is exactly `sl` long. TUL's actual object is an entire
VARIABLE-length released trajectory, so this module needs real sequence
machinery `reid.py` never does (padding, a Transformer key-padding mask,
`pack_padded_sequence` for the LSTM) -- and the long-term augmentation
trick above is meaningless for a fixed-length window (there's no notion of
"a longer version of this window" the way there is for "this user's other
trajectories"). The distinction is real, not cosmetic.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def build_trajectory_dataset(df: pd.DataFrame, max_traj_len: int = 150, min_traj_len: int = 3):
    """Groups the unified parquet into whole trajectories ((user_id, traj_id)
    groups, sorted by `ts`, truncated to `max_traj_len` points), split into
    train/test by relabeling `mi_split` ("member"->train, "nonmember"->test)
    -- see module docstring for why this reuses membership_split instead of
    MainTUL's own 80/20 split, and why `user_disjoint` users are dropped
    entirely (closed-world classifier).

    Returns dict with:
      train / test: {raw_user_id: [(cell_arr, hour_arr), ...]} -- `cell_arr`
        is the (1-indexed) cell_id sequence; `hour_arr` is hour-of-day
        SHIFTED +1 (1..24) so 0 is free as this module's pad value.
      user_to_idx: {raw_user_id: 0-indexed class label}, only users with
        >=1 TRAIN trajectory (a user with zero train trajectories can never
        be predicted correctly by a closed-world classifier, so they
        contribute no class and no test rows here).
      total_locations: max cell_id in the whole (unfiltered) dataframe --
        sizes the POI embedding table the same way puffin.models.mopae_torch
        does.
    """
    use = df.loc[~df["user_disjoint"], ["user_id", "traj_id", "ts", "cell_id", "hour", "mi_split"]]
    trajs_train: dict = {}
    trajs_test: dict = {}
    for (user_id, _traj_id), g in use.groupby(["user_id", "traj_id"], sort=False):
        if len(g) < min_traj_len:
            continue
        g = g.sort_values("ts")
        cell = g["cell_id"].to_numpy()[:max_traj_len].astype(np.int64)
        hour = (g["hour"].to_numpy()[:max_traj_len].astype(np.int64) + 1)  # shift: 0 reserved for pad
        split = g["mi_split"].iat[0]  # constant within a traj_id group by construction (membership_split)
        bucket = trajs_train if split == "member" else trajs_test
        bucket.setdefault(int(user_id), []).append((cell, hour))

    user_ids = sorted(u for u in trajs_train if len(trajs_train[u]) >= 1)
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    train = {u: trajs_train[u] for u in user_ids}
    test = {u: trajs_test.get(u, []) for u in user_ids}
    total_locations = int(df["cell_id"].max())
    return dict(train=train, test=test, user_to_idx=user_to_idx, total_locations=total_locations)


def _flatten(split_dict: dict, user_to_idx: dict):
    """[(user_class, raw_user_id, idx_within_user, cell_arr, hour_arr), ...]
    -- `idx_within_user` is needed so long-term augmentation can exclude
    "this" trajectory when sampling others for the same user."""
    items = []
    for u, trajs in split_dict.items():
        if u not in user_to_idx:
            continue
        for i, (cell, hour) in enumerate(trajs):
            items.append((user_to_idx[u], u, i, cell, hour))
    return items


def _sample_long_term(train_trajs_for_user, this_idx: int, long_term_num: int, rng: np.random.Generator):
    """Concatenates "this" trajectory with up to `long_term_num` other
    randomly sampled trajectories of the SAME user, drawn from that user's
    full train pool (see module docstring: matches the original's own
    `long_term_augment`, which also samples irrespective of any
    train/validation subsplit). Clamped to `n_train_trajs - 1` so a user
    with few trajectories degrades gracefully (fewer/no extra context)
    instead of crashing like the original (`random.sample` on a
    population smaller than `long_term_num` raises `ValueError`)."""
    n = len(train_trajs_for_user)
    k = min(long_term_num, max(0, n - 1))
    if k > 0:
        others = [i for i in range(n) if i != this_idx]
        chosen = rng.choice(others, size=k, replace=False).tolist()
    else:
        chosen = []
    idxs = sorted(set(chosen + [this_idx]))
    cells = np.concatenate([train_trajs_for_user[i][0] for i in idxs])
    hours = np.concatenate([train_trajs_for_user[i][1] for i in idxs])
    return cells, hours


def _pad_batch(cell_list, hour_list, device):
    lens = np.array([len(c) for c in cell_list], dtype=np.int64)
    maxlen = int(lens.max())
    cell_pad = np.zeros((len(cell_list), maxlen), dtype=np.int64)
    hour_pad = np.zeros((len(cell_list), maxlen), dtype=np.int64)
    for i, (c, h) in enumerate(zip(cell_list, hour_list)):
        cell_pad[i, :len(c)] = c
        hour_pad[i, :len(h)] = h
    return (torch.tensor(cell_pad, device=device), torch.tensor(hour_pad, device=device),
            torch.tensor(lens, device=device))


class _Embedding(nn.Module):
    """POI id + hour-of-day embedding, fused via a small MLP -- same shape
    as MainTUL's LstmTimeAwareEmbedding/TransformerTimeAwareEmbedding with
    the category view removed (see module docstring)."""

    def __init__(self, total_locations: int, embed_size: int, hour_vocab: int = 25):
        super().__init__()
        self.poi_embed = nn.Embedding(total_locations + 1, embed_size, padding_idx=0)
        self.hour_embed = nn.Embedding(hour_vocab, embed_size // 4, padding_idx=0)
        self.fc = nn.Linear(embed_size + embed_size // 4, embed_size)
        self.dropout = nn.Dropout(0.1)
        self.out_features = embed_size

    def forward(self, cell, hour):
        emb = torch.cat([self.poi_embed(cell), self.hour_embed(hour)], dim=-1)
        return self.dropout(torch.tanh(self.fc(emb)))


class StudentEncoder(nn.Module):
    """Lightweight LSTM -> last real hidden state -> linear classifier.
    This is the network actually meant to be deployed as the attacker
    after distillation: MainTUL's whole point is that the cheap student
    ends up as accurate as the Transformer teacher."""

    def __init__(self, embedding: _Embedding, hidden_size: int, n_users: int):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=embedding.out_features, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_users)

    def forward(self, cell, hour, lengths):
        emb = self.embedding(cell, hour)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _c) = self.lstm(packed)
        return self.classifier(h[-1])


class _TemporalPositionalEncoding(nn.Module):
    """MainTUL's learnable temporal encoding, `cos(position * w + b)` per
    embedding dim -- see module docstring for why `position` here is a
    0-indexed within-sequence index rather than a raw unix timestamp."""

    def __init__(self, embed_size: int):
        super().__init__()
        self.w = nn.Parameter((1.0 / 10 ** torch.linspace(0, 9, embed_size)).float(), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(embed_size), requires_grad=True)
        self.scale = math.sqrt(1.0 / embed_size)

    def forward(self, seq_len: int, device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device).float().unsqueeze(-1)
        return self.scale * torch.cos(pos * self.w.unsqueeze(0) + self.b.unsqueeze(0))  # (seq_len, embed_size)


class TeacherEncoder(nn.Module):
    """Transformer encoder over the (possibly long-term-augmented)
    sequence, masked max-pooled over real (non-pad) positions, then a
    linear classifier -- same shape as MainTUL's TeacherEncoder with the
    category view removed."""

    def __init__(self, embedding: _Embedding, embed_size: int, hidden_size: int, n_users: int,
                 num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.embedding = embedding
        self.pos_encoding = _TemporalPositionalEncoding(embed_size)
        layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(embed_size))
        self.classifier = nn.Linear(embed_size, n_users)

    def forward(self, cell, hour, lengths):
        pad_mask = cell == 0  # cell_id is 1-indexed in this project's schema, so 0 is always padding
        emb = self.embedding(cell, hour) + self.pos_encoding(cell.shape[1], cell.device).unsqueeze(0)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        weights = torch.zeros_like(out)
        weights.masked_fill_(pad_mask.unsqueeze(-1), -1e6)
        pooled, _ = torch.max(out + weights, dim=1)
        return self.classifier(pooled)


def _kl_distill(student_logits, teacher_logits, temperature: float):
    """Bidirectional distillation term: KL(student || teacher) in softened
    probability space, scaled by temperature^2 (standard Hinton-style
    distillation scaling, same as MainTUL's `compute_loss_dis`).
    `reduction="batchmean"` replaces the original's deprecated
    `size_average=False` (identical computation, no deprecation warning)."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def _per_traj_predictions(logits: torch.Tensor, labels: torch.Tensor):
    ranked = torch.argsort(logits, dim=-1, descending=True)
    top1 = (ranked[:, 0] == labels).cpu().numpy()
    top5 = (ranked[:, :5] == labels.unsqueeze(1)).any(dim=1).cpu().numpy()
    conf = F.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
    return top1, top5, conf


def run_tul_maintul(df: pd.DataFrame, device, max_traj_len: int = 150, min_traj_len: int = 3,
                     long_term_num: int = 8, embed_size: int = 128, hidden_size: int = 256,
                     num_layers: int = 2, num_heads: int = 4, batch_size: int = 128,
                     max_epochs: int = 30, patience: int = 5, lr: float = 1e-3,
                     temperature: float = 10.0, lambda_distill: float = 10.0,
                     val_frac: float = 0.1, seed: int = 0):
    """Phase 3 task 4's TUL attack: trains the ported MainTUL student/
    teacher pair (see module docstring) on GeoLife's own trajectories
    (stand-in "released trajectories" -- no TrajGAN/GI-DP baseline exists
    yet to attack for real) and reports both standalone attackers' test
    performance.

    Early stopping is on the STUDENT's standalone validation top-1
    accuracy (a random `val_frac` of train trajectories, held out from
    gradient updates but still eligible to be sampled as another training
    trajectory's long-term-augmentation partner -- see module docstring:
    the original has the same property, not a new leak introduced here)
    -- the student is the one actually meant to be deployed as the
    attacker, so that's what early stopping should protect.

    Returns (per_user_df, summary):
      per_user_df: one row per user with >=1 train trajectory, columns
        user_id, n_train_trajectories, n_test_trajectories,
        student_top1, student_top5, teacher_top1, teacher_top5 (mean over
        that user's TEST trajectories; NaN if the user has none),
        student_mean_confidence, teacher_mean_confidence -- per-user, not
        just an aggregate, matching this project's whole per-user-risk
        methodology (puffin.attacks.reid's convention).
      summary: dict with n_users, n_train_trajectories, n_val_trajectories,
        n_test_trajectories, student_overall_top1/top5,
        teacher_overall_top1/top5, and the hyperparameters used.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = build_trajectory_dataset(df, max_traj_len=max_traj_len, min_traj_len=min_traj_len)
    user_to_idx = data["user_to_idx"]
    n_users = len(user_to_idx)
    idx_to_raw = {i: u for u, i in user_to_idx.items()}

    train_items = _flatten(data["train"], user_to_idx)
    test_items = _flatten(data["test"], user_to_idx)

    val_mask = rng.random(len(train_items)) < val_frac
    val_items = [it for it, m in zip(train_items, val_mask) if m]
    fit_items = [it for it, m in zip(train_items, val_mask) if not m]

    embedding = _Embedding(data["total_locations"], embed_size).to(device)
    student = StudentEncoder(embedding, hidden_size, n_users).to(device)
    teacher_embedding = _Embedding(data["total_locations"], embed_size).to(device)
    teacher = TeacherEncoder(teacher_embedding, embed_size, hidden_size, n_users,
                              num_layers=num_layers, num_heads=num_heads).to(device)
    params = list(student.parameters()) + list(teacher.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    def make_batch(items, augment: bool):
        labels = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
        cur_cell, cur_hour = [it[3] for it in items], [it[4] for it in items]
        current = _pad_batch(cur_cell, cur_hour, device)
        if not augment:
            return current, current, labels
        lt_cell, lt_hour = [], []
        for _cls, raw_u, idx_in_user, _c, _h in items:
            c, h = _sample_long_term(data["train"][raw_u], idx_in_user, long_term_num, rng)
            lt_cell.append(c)
            lt_hour.append(h)
        longterm = _pad_batch(lt_cell, lt_hour, device)
        return current, longterm, labels

    def student_top1_acc(items):
        if not items:
            return 0.0
        student.eval()
        with torch.no_grad():
            (cell, hour, lens), _, labels = make_batch(items, augment=False)
            logits = student(cell, hour, lens)
        return float((logits.argmax(-1) == labels).float().mean().item())

    best_val_acc, best_state, bad_epochs = -1.0, None, 0
    for _epoch in range(max_epochs):
        student.train()
        teacher.train()
        perm = rng.permutation(len(fit_items))
        for start in range(0, len(perm), batch_size):
            batch = [fit_items[i] for i in perm[start:start + batch_size]]
            (cur_cell, cur_hour, cur_len), (lt_cell, lt_hour, lt_len), labels = make_batch(batch, augment=True)

            # Pass 1: student sees the short view, teacher sees the long-term view.
            student_out = student(cur_cell, cur_hour, cur_len)
            teacher_out = teacher(lt_cell, lt_hour, lt_len)
            loss1 = (F.cross_entropy(student_out, labels) + F.cross_entropy(teacher_out, labels)
                     + lambda_distill * _kl_distill(student_out, teacher_out, temperature))

            # Pass 2 (role-swapped): student sees the long-term view, teacher sees the short view.
            student_out2 = student(lt_cell, lt_hour, lt_len)
            teacher_out2 = teacher(cur_cell, cur_hour, cur_len)
            loss2 = (F.cross_entropy(student_out2, labels) + F.cross_entropy(teacher_out2, labels)
                     + lambda_distill * _kl_distill(teacher_out2, student_out2, temperature))

            opt.zero_grad()
            (loss1 + loss2).backward()
            opt.step()

        val_acc = student_top1_acc(val_items) if val_items else student_top1_acc(fit_items[:batch_size])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (copy.deepcopy(student.state_dict()), copy.deepcopy(teacher.state_dict()))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        student.load_state_dict(best_state[0])
        teacher.load_state_dict(best_state[1])
    student.eval()
    teacher.eval()

    rows_by_user: dict = {idx_to_raw[i]: dict(
        user_id=idx_to_raw[i],
        n_train_trajectories=len(data["train"][idx_to_raw[i]]),
        n_test_trajectories=len(data["test"][idx_to_raw[i]]),
        student_top1=[], student_top5=[], teacher_top1=[], teacher_top5=[],
        student_conf=[], teacher_conf=[],
    ) for i in range(n_users)}

    all_student_top1, all_student_top5, all_teacher_top1, all_teacher_top5 = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(test_items), batch_size):
            batch = test_items[start:start + batch_size]
            (cell, hour, lens), _, labels = make_batch(batch, augment=False)
            s_logits = student(cell, hour, lens)
            t_logits = teacher(cell, hour, lens)
            s_top1, s_top5, s_conf = _per_traj_predictions(s_logits, labels)
            t_top1, t_top5, t_conf = _per_traj_predictions(t_logits, labels)
            all_student_top1.append(s_top1); all_student_top5.append(s_top5)
            all_teacher_top1.append(t_top1); all_teacher_top5.append(t_top5)
            for (cls, raw_u, *_), st1, st5, sc, tt1, tt5, tc in zip(
                    batch, s_top1, s_top5, s_conf, t_top1, t_top5, t_conf):
                row = rows_by_user[raw_u]
                row["student_top1"].append(bool(st1)); row["student_top5"].append(bool(st5))
                row["teacher_top1"].append(bool(tt1)); row["teacher_top5"].append(bool(tt5))
                row["student_conf"].append(float(sc)); row["teacher_conf"].append(float(tc))

    rows = []
    for raw_u, row in rows_by_user.items():
        n_test = row["n_test_trajectories"]
        rows.append(dict(
            user_id=raw_u,
            n_train_trajectories=row["n_train_trajectories"],
            n_test_trajectories=n_test,
            student_top1=float(np.mean(row["student_top1"])) if n_test else float("nan"),
            student_top5=float(np.mean(row["student_top5"])) if n_test else float("nan"),
            teacher_top1=float(np.mean(row["teacher_top1"])) if n_test else float("nan"),
            teacher_top5=float(np.mean(row["teacher_top5"])) if n_test else float("nan"),
            student_mean_confidence=float(np.mean(row["student_conf"])) if n_test else float("nan"),
            teacher_mean_confidence=float(np.mean(row["teacher_conf"])) if n_test else float("nan"),
        ))
    per_user_df = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)

    def _cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)

    summary = dict(
        n_users=n_users,
        n_train_trajectories=len(fit_items),
        n_val_trajectories=len(val_items),
        n_test_trajectories=len(test_items),
        student_overall_top1=float(_cat(all_student_top1).mean()) if test_items else float("nan"),
        student_overall_top5=float(_cat(all_student_top5).mean()) if test_items else float("nan"),
        teacher_overall_top1=float(_cat(all_teacher_top1).mean()) if test_items else float("nan"),
        teacher_overall_top5=float(_cat(all_teacher_top5).mean()) if test_items else float("nan"),
        best_val_acc=best_val_acc,
        hyperparameters=dict(max_traj_len=max_traj_len, min_traj_len=min_traj_len,
                              long_term_num=long_term_num, embed_size=embed_size,
                              hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                              batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                              lr=lr, temperature=temperature, lambda_distill=lambda_distill,
                              val_frac=val_frac, seed=seed),
    )
    return per_user_df, summary
