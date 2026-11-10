"""Phase 5: TrajGAN baseline -- synthetic-trajectory release mechanism.

Per `puffin/attacks/README.md`'s threat-model table, TrajGAN releases
SYNTHETIC TRAJECTORIES (not encoder features like Mo-PAE/PUFFIN), so it is
attacked with TUL (`puffin/attacks/tul_maintul.py`) and MIA, never the
feature-based attacks. This module builds the mechanism itself; running the
attack suite against it is a later phase's job (per the task's own scope).

## What "TrajGAN" means here, and the repo it was ported from

"TrajGAN" canonically refers to Rao, Gao, Kang & Huang's **LSTM-TrajGAN**
(GIScience 2021, arXiv:2006.10521), the well-known LSTM+GAN trajectory
generator with an explicit privacy framing (evaluated via TUL in the
original paper, same attack family this project already uses). Verified
this is the right paper/repo before touching anything: the abstract, the
architecture description, and the TUL-based evaluation protocol all match.
Official code: https://github.com/GeoDS/LSTM-TrajGAN, cloned to
`legacy/LSTM-TrajGAN/` (not committed, same convention as every other
third-party clone in this project -- `.gitignore` updated the same way).

**Unlike POIPrivacy and genuinely unlike even MainTUL, LSTM-TrajGAN's own
demo ran essentially out of the box.** Built an isolated Python 3.6 venv
(`legacy/LSTM-TrajGAN/.venv`, this project's own `.venv` untouched) with
its exact pinned requirements (numpy==1.18.4, pandas==1.1.5,
tensorflow==1.13.1 -- CPU build, not tensorflow-gpu, since erebor's driver
only supports CUDA 12.x and TF 1.13 needs CUDA 10.0; not an issue since the
repo's own demo is tiny and CPU-fast anyway -- Keras==2.2.4, geohash2==1.1,
scikit-learn==0.23.2). `python train.py 20 256 10` (20 epochs, their own
bundled Foursquare-NYC-derived demo data in `data/final_train.npy`) ran to
completion with zero code changes: D loss fell 32.3 -> 8.2, G loss fell
2094 -> 1986 over 20 epochs -- a sane, non-exploding adversarial training
curve. The ONE bug found was in `predict.py` (generation step, not
training): `keras.engine.saving.load_weights_from_hdf5_group` does
`f.attrs['keras_version'].decode('utf8')`, assuming h5py's old
bytes-valued attrs API; h5py>=3.0 (the version pip resolves by default
for Keras 2.2.4's unpinned `h5py` dependency today) stores attrs as `str`
already, which has no `.decode` -> `AttributeError` on the very first
`load_weights` call. Fixed by pinning `h5py==2.10.0` (the version
contemporaneous with Keras 2.2.4/TF 1.13.1); `predict.py 20` then ran
cleanly and produced a real `results/syn_traj_test.csv` (22,153 synthetic
check-in rows). This is a plain dependency-version bug, same class as
MainTUL's `np.Inf`/logging-format bugs, not a structural blocker.

**Because the original ran cleanly, this module PORTS (does not import --
the clone isn't committed) its real generator/discriminator/TrajLoss
architecture into this project's schema and PyTorch**, same policy as
`puffin/attacks/tul_maintul.py` for MainTUL. Read directly from the
original's `model.py`/`losses.py`, not reconstructed from the paper alone.

## Port decisions (what changed vs. the original, and why)

- **`category` field/output head dropped entirely.** The original treats
  trajectory points as (lat, lon, day, hour, **category**) with category a
  10-class one-hot venue-type label straight from Foursquare's POI
  metadata. This project's unified schema (`puffin/data/schema.py`) has no
  category column at all -- a GeoLife GPS fix isn't a venue with a type,
  and even Foursquare's own loader here drops category because the schema
  has nowhere to put it. Identical reasoning, independently arrived at, to
  `tul_maintul.py`'s POI-category drop and `mia_inhouse.py`/POIPrivacy's
  writeup. Dropping it removes one embedding branch, one output head, and
  one term from TrajLoss (`ce_category_mean*p_cat`); everything else
  (lat_lon, day/weekday, hour) ports unchanged in kind.
- **Synthetic `cell_id` is NOT a GAN output.** The original only ever
  outputs continuous (lat, lon) -- there is no discrete location-id output
  to port, because the original's discrete output (`category`) is the
  field this port drops. But `tul_maintul.py` (the TUL attack meant to run
  against this mechanism's output, per the threat-model table) consumes
  `cell_id`, not raw lat/lon. So after generation, this module re-
  tessellates the synthetic (lat, lon) pairs with
  `puffin.data.tessellation.assign_grid_cells` (same function Phase 1 used
  to build every real dataset's `cell_id`) to produce a dense, 1-indexed,
  schema-valid `cell_id` for the synthetic trajectories. This is a new
  step with no analogue in the original (which only ever had to emit a csv
  for a bespoke downstream MARC/TUL script), needed purely to make the
  output a drop-in unified-schema dataframe the rest of this project's
  attack code can consume unchanged.
- **Trajectories truncated to `max_traj_len`** (default 150, same default
  as `tul_maintul.py`, for the same reason): GeoLife's real per-trip
  trajectories run up to 92,645 points
  (`data/processed/geolife/unified.parquet`), far past the original's own
  `max_length=144` (itself sized for check-in trajectories, tens of
  points/day) and far past what an LSTM over the full sequence could
  train on in any reasonable time. Each trajectory keeps its first
  `max_traj_len` chronological points; pre-padding (original's own
  `pad_sequences(..., padding='pre')` convention, kept as-is) and a
  boolean mask handle the rest, exactly as the original.
- **`scale_factor` uses a robust percentile, not the original's true max.**
  GeoLife's raw GPS logs contain a handful of physically-impossible fixes
  (lat up to ~400 degrees seen in `data/processed/geolife/unified.parquet`);
  the original's `scale_factor = max(abs deviation)` would let one such
  glitch dominate the entire normalization. See `_latlon_centroid_scale`'s
  docstring for the full reasoning -- same class of judgment call as
  `tul_maintul.py`'s length truncation, not a correctness fix to the
  original (which never had to contend with data this noisy).
- **This project's own `split` column reused** (`puffin.data.splits.
  temporal_split`: per-user chronological 80/20) in place of re-deriving
  the original's own ad hoc train/test split -- same "reuse existing infra
  instead of duplicating it" pattern `tul_maintul.py`/`mia_inhouse.py`
  already established. The generator is trained on `split=="train"`
  trajectories and generates synthetic counterparts of `split=="test"`
  trajectories, mirroring the original's own train.py/predict.py split
  between training data and the data actually released as synthetic.
- **Training loop uses the "regular" full-epoch-over-all-batches loop**,
  not the original's *default* loop (one random batch sampled per "epoch,"
  repeated `n_epochs` times -- the original's own train.py has this
  alternative loop commented out, with a code comment from the authors
  themselves noting it "usually takes fewer epochs to converge" than the
  default). Since this port's job is to produce a real, sane baseline on
  our data rather than reproduce the original's exact epoch-counting
  convention, the better-documented alternative is the one ported.
- **Not ported**: `geohash2`-based CSV encoding (irrelevant -- this project
  tessellates into `cell_id` instead of geohashing), the bundled
  MARC-based `TUL_test.py` evaluation (this project already has its own
  TUL attack, `tul_maintul.py`, which is what the threat-model table names
  for this mechanism -- running the original's bespoke MARC classifier
  would be a second, redundant TUL implementation), per-epoch file
  checkpointing to disk (kept only in-memory; this module returns the
  trained generator's final-epoch weights implicitly via the synthetic
  output, not a saved checkpoint file -- GANs have no clean "best epoch by
  validation accuracy" criterion the way `tul_maintul.py`/`reid.py`'s
  early stopping does).

## Validation on real project data

See `scripts/smoke_test_phase5_trajgan.py` (committed, same convention as
`scripts/smoke_test_phase3_tul.py`) for the real GeoLife run and numbers
reported in `CHANGES_FROM_ORIGINAL.md`/`reports/05_baselines_trajgan_gidp.md`.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from puffin.data.tessellation import assign_grid_cells

MASK_PAD_VALUE = 0.0


# --------------------------------------------------------------------------
# Data prep: real unified-schema trajectories -> fixed-length padded tensors
# --------------------------------------------------------------------------

def _trajectory_groups(df: pd.DataFrame, split_name: str, max_traj_len: int, min_traj_len: int):
    """One (lat, lon, weekday, hour, length) tuple per (user_id, traj_id)
    trajectory in the given split, sorted chronologically, truncated to
    the first `max_traj_len` points. Trajectories shorter than
    `min_traj_len` are dropped (nothing useful to generate or attack)."""
    sub = df[df["split"] == split_name].sort_values(["user_id", "traj_id", "ts"])
    # PI-caught performance bug (2026-11-09, live py-spy stack trace):
    # pandas' NDFrame.__finalize__ deepcopies a non-empty .attrs dict on
    # EVERY slice/groupby operation. df carries assign_grid_cells's
    # grid_cell_mapping (tens of thousands of entries) in .attrs, so every
    # .iloc[] call inside this loop paid a full deepcopy of that dict --
    # the actual cause of TrajGAN's original 7.2h/2-epoch "slowdown"
    # (nothing to do with the GPU or the GAN itself). Clearing attrs here
    # makes every downstream groupby/slice a no-op check (`if other.attrs`)
    # instead of a deepcopy.
    sub.attrs = {}
    out = []
    for (user_id, traj_id), g in sub.groupby(["user_id", "traj_id"], sort=False):
        if len(g) < min_traj_len:
            continue
        g = g.iloc[:max_traj_len]
        out.append(dict(
            user_id=int(user_id), traj_id=int(traj_id),
            lat=g["lat"].to_numpy(dtype=np.float64),
            lon=g["lon"].to_numpy(dtype=np.float64),
            weekday=g["weekday"].to_numpy(dtype=np.int64),
            hour=g["hour"].to_numpy(dtype=np.int64),
            ts=g["ts"].to_numpy(),
            length=len(g),
        ))
    return out


def _trajectory_chunks(df: pd.DataFrame, split_name: str, max_traj_len: int, min_traj_len: int = 1,
                        group_by: str = "trajectory"):
    """Chunks every trajectory-like group in the split into consecutive
    pieces of AT MOST max_traj_len points -- never truncates a group's
    tail away. `min_traj_len` filters at the WHOLE-GROUP level: a group
    shorter than this is dropped entirely before chunking, not per-chunk.
    Called with `min_traj_len=1` (no filtering) for TEST, so every real
    test point is covered by exactly one chunk (fixes the PI-caught
    window-alignment bug, 2026-11-09: truncating to the first
    max_traj_len points silently dropped everything past that for any
    longer trajectory). Called with the project's normal min_traj_len for
    TRAIN too (PI instruction, 2026-11-09): the generator must train on
    the same chunking it generates from, not truncated trajectory
    openings.

    `group_by="trajectory"` (default, GeoLife): groups by the dataset's
    own (user_id, traj_id) -- a real separate trip.
    `group_by="week"` (Foursquare, Gowalla; PI instruction 2026-11-09,
    following the original LSTM-TrajGAN paper's per-user weekly
    trajectory definition, Rao et al. 2020): groups by (user_id,
    ISO-year, ISO-week) instead -- these datasets' own `traj_id` is only
    a same-calendar-day grouping invented for check-in data with no
    native trajectory id, which leaves almost every "trajectory" shorter
    than any reasonable min_traj_len (the same issue already fixed for
    the rest of this project's windowing via respect_traj_boundary=False,
    which TrajGAN's per-trajectory chunking design has no equivalent of).

    Each chunk keeps the parent group's user_id/traj_id (week mode: the
    FIRST real traj_id in that (user, week) group, for bookkeeping only)
    -- `ts` is enough to reconstruct point-for-point identity with the
    real data, chunk membership itself is never inspected downstream."""
    sub = df[df["split"] == split_name].copy()
    sub.attrs = {}  # see windowing.py's comment -- same fix, same root cause
    if group_by == "week":
        iso = pd.to_datetime(sub["ts"]).dt.isocalendar()
        sub["_group_key"] = sub["user_id"].astype(str) + "_" + iso.year.astype(str) + "_W" + iso.week.astype(str)
        sub = sub.sort_values(["user_id", "_group_key", "ts"])
        group_cols = ["user_id", "_group_key"]
    else:
        sub = sub.sort_values(["user_id", "traj_id", "ts"])
        group_cols = ["user_id", "traj_id"]

    out = []
    for key, g in sub.groupby(group_cols, sort=False):
        user_id = key[0]
        traj_id = int(g["traj_id"].iloc[0])
        n = len(g)
        if n < min_traj_len:
            continue
        for chunk_start in range(0, n, max_traj_len):
            chunk = g.iloc[chunk_start:chunk_start + max_traj_len]
            out.append(dict(
                user_id=int(user_id), traj_id=traj_id,
                lat=chunk["lat"].to_numpy(dtype=np.float64),
                lon=chunk["lon"].to_numpy(dtype=np.float64),
                weekday=chunk["weekday"].to_numpy(dtype=np.int64),
                hour=chunk["hour"].to_numpy(dtype=np.int64),
                ts=chunk["ts"].to_numpy(),
                length=len(chunk),
            ))
    return out


def _latlon_centroid_scale(train_groups, test_groups):
    """Centroid computed over train+test combined, exactly as the
    original's train.py/predict.py (`lat_centroid`/`lon_centroid`) -- a
    normalization constant shared so a trajectory generated at test time
    uses the same coordinate frame as training, not a privacy-relevant
    statistic.

    `scale_factor` deliberately deviates from the original here: the
    original takes the TRUE max absolute deviation from centroid. On this
    project's real GeoLife data that is badly outlier-dominated -- GeoLife's
    raw GPS logs contain a small number of physically-impossible fixes
    (`lat` up to ~400 degrees, `lon` spanning the full -180..180 range
    found in `data/processed/geolife/unified.parquet`, a known raw-data
    noise characteristic of this dataset, not introduced by this port). A
    true max would set `scale_factor` to whatever that single worst glitch
    is, which both wastes nearly all of the tanh output's dynamic range on
    points that will never recur and makes `mean_abs_lat/lon_error` (the
    utility proxy this module reports) report in units dominated by that
    one outlier rather than typical trajectory precision. Using the 99.5th
    percentile of absolute deviation instead (and clipping the *input*
    normalization to [-1, 1], so an outlier point saturates rather than
    silently rescaling everything) keeps the original's own mechanism
    (bounded tanh output * scale_factor) intact while making it usable on
    real, noisy mobility data -- the same kind of "necessary, documented
    adaptation, not a correctness bug" judgment call as `tul_maintul.py`'s
    `max_traj_len` truncation."""
    all_lat = np.concatenate([g["lat"] for g in train_groups + test_groups])
    all_lon = np.concatenate([g["lon"] for g in train_groups + test_groups])
    lat_c, lon_c = float(all_lat.mean()), float(all_lon.mean())
    scale = max(np.percentile(np.abs(all_lat - lat_c), 99.5),
                np.percentile(np.abs(all_lon - lon_c), 99.5))
    return lat_c, lon_c, float(max(scale, 1e-6))


def _pad_batch(groups, idx, max_traj_len, lat_c, lon_c, scale):
    """Pre-pad (zeros at the front, real points at the end -- original's
    own `pad_sequences(..., padding='pre')` convention) a list of
    trajectories (indexed by `idx` into `groups`) into fixed-length
    tensors. Returns latlon (B,L,2) normalized, weekday_oh (B,L,7),
    hour_oh (B,L,24), mask (B,L) bool (True at real, non-padded
    positions), lengths (B,)."""
    B = len(idx)
    latlon = np.zeros((B, max_traj_len, 2), dtype=np.float32)
    weekday_oh = np.zeros((B, max_traj_len, 7), dtype=np.float32)
    hour_oh = np.zeros((B, max_traj_len, 24), dtype=np.float32)
    mask = np.zeros((B, max_traj_len), dtype=np.float32)
    lengths = np.zeros(B, dtype=np.int64)
    for b, i in enumerate(idx):
        g = groups[i]
        L = g["length"]
        start = max_traj_len - L
        # Clip to [-1, 1]: an outlier point beyond the 99.5th-percentile
        # scale (see _latlon_centroid_scale) saturates instead of blowing
        # up the whole batch's normalized magnitude.
        latlon[b, start:, 0] = np.clip((g["lat"] - lat_c) / scale, -1.0, 1.0)
        latlon[b, start:, 1] = np.clip((g["lon"] - lon_c) / scale, -1.0, 1.0)
        weekday_oh[b, np.arange(start, max_traj_len), g["weekday"].clip(0, 6)] = 1.0
        hour_oh[b, np.arange(start, max_traj_len), g["hour"].clip(0, 23)] = 1.0
        mask[b, start:] = 1.0
        lengths[b] = L
    return (torch.from_numpy(latlon), torch.from_numpy(weekday_oh),
            torch.from_numpy(hour_oh), torch.from_numpy(mask), lengths)


# --------------------------------------------------------------------------
# Model: ported from model.py (Keras functional API -> plain PyTorch)
# --------------------------------------------------------------------------

class TrajGANGenerator(nn.Module):
    """Takes a real (lat_lon, weekday, hour) trajectory + noise, outputs a
    synthetic trajectory of the same shape. Mirrors the original's
    `build_generator`: a per-timestep Dense "embedding" for each field
    (nn.Linear already broadcasts over the (B, L, *) leading dims, so the
    original's unstack/Dense/stack dance isn't needed in PyTorch), fused
    with a *per-timestep-repeated* noise vector (same noise draw at every
    timestep, exactly as the original's `Concatenate([x, noise])` inside
    the per-timestep loop), then a many-to-many LSTM and three output
    heads (lat_lon via tanh*scale_factor, weekday/hour via softmax)."""

    def __init__(self, latent_dim: int = 100, lstm_units: int = 100):
        super().__init__()
        self.emb_latlon = nn.Linear(2, 64)
        self.emb_weekday = nn.Linear(7, 7)
        self.emb_hour = nn.Linear(24, 24)
        self.fuse = nn.Linear(64 + 7 + 24 + latent_dim, 100)
        self.lstm = nn.LSTM(100, lstm_units, batch_first=True)
        self.out_latlon = nn.Linear(lstm_units, 2)
        self.out_weekday = nn.Linear(lstm_units, 7)
        self.out_hour = nn.Linear(lstm_units, 24)

    def forward(self, latlon, weekday_oh, hour_oh, noise, scale_factor: float):
        B, L, _ = latlon.shape
        e = torch.cat([
            F.relu(self.emb_latlon(latlon)),
            F.relu(self.emb_weekday(weekday_oh)),
            F.relu(self.emb_hour(hour_oh)),
        ], dim=-1)
        noise_rep = noise.unsqueeze(1).expand(-1, L, -1)
        fused = F.relu(self.fuse(torch.cat([e, noise_rep], dim=-1)))
        lstm_out, _ = self.lstm(fused)
        gen_latlon = torch.tanh(self.out_latlon(lstm_out)) * scale_factor
        gen_weekday = F.softmax(self.out_weekday(lstm_out), dim=-1)
        gen_hour = F.softmax(self.out_hour(lstm_out), dim=-1)
        return gen_latlon, gen_weekday, gen_hour


class TrajGANDiscriminator(nn.Module):
    """Real-vs-synthetic classifier. Mirrors `build_discriminator`: same
    per-field embeddings (no noise, no mask), concatenated, many-to-ONE
    LSTM (final hidden state), sigmoid. Padding is pre-padded zeros for
    both real and fake inputs identically, so -- as in the original -- the
    discriminator is never given an explicit mask and simply learns the
    padding pattern is uninformative."""

    def __init__(self, lstm_units: int = 100):
        super().__init__()
        self.emb_latlon = nn.Linear(2, 64)
        self.emb_weekday = nn.Linear(7, 7)
        self.emb_hour = nn.Linear(24, 24)
        self.fuse = nn.Linear(64 + 7 + 24, 100)
        self.lstm = nn.LSTM(100, lstm_units, batch_first=True)
        self.out = nn.Linear(lstm_units, 1)

    def forward(self, latlon, weekday_oh, hour_oh):
        e = torch.cat([
            F.relu(self.emb_latlon(latlon)),
            F.relu(self.emb_weekday(weekday_oh)),
            F.relu(self.emb_hour(hour_oh)),
        ], dim=-1)
        fused = F.relu(self.fuse(e))
        _, (h_n, _) = self.lstm(fused)
        return torch.sigmoid(self.out(h_n[-1])).squeeze(-1)


def _traj_loss(mask, lengths, real_latlon, gen_latlon, real_weekday_oh, gen_weekday,
               real_hour_oh, gen_hour, d_pred_fake, p_bce=1.0, p_latlon=10.0,
               p_weekday=1.0, p_hour=1.0):
    """Ported from `losses.py`'s `trajLoss`: adversarial BCE (fool D) plus
    masked reconstruction terms (MSE for lat_lon, cross-entropy for
    weekday/hour), each divided by the trajectory's true length so short
    and long trajectories contribute comparably -- same normalization the
    original applies via `traj_length`. `category`'s CE term is dropped
    (see module docstring)."""
    bce = F.binary_cross_entropy(d_pred_fake, torch.ones_like(d_pred_fake))

    len_f = lengths.to(real_latlon.dtype).clamp(min=1)
    sq_err = ((gen_latlon - real_latlon) ** 2).sum(dim=-1)  # (B, L)
    masked_latlon_mse = ((sq_err * mask).sum(dim=1) / len_f).sum()

    def masked_ce(gen_probs, real_oh):
        ce = -(real_oh * torch.log(gen_probs.clamp_min(1e-9))).sum(dim=-1)  # (B, L)
        return ((ce * mask).sum(dim=1) / len_f).sum()

    ce_weekday = masked_ce(gen_weekday, real_weekday_oh)
    ce_hour = masked_ce(gen_hour, real_hour_oh)

    return bce * p_bce + masked_latlon_mse * p_latlon + ce_weekday * p_weekday + ce_hour * p_hour


def generate_from_generator(G, test_groups, max_traj_len, latent_dim, lat_c, lon_c, scale, device,
                             batch_size=64):
    """The generation half of run_trajgan, factored out so the driver can
    call it again with a DIFFERENT G (e.g. rebuilt from an earlier
    checkpoint, per the PI's collapse-handling instruction, 2026-11-09)
    without retraining. Returns (synthetic_df, summary_fields_dict,
    generation_time)."""
    import time as _time
    G.eval()
    rows = []
    abs_lat_errs, abs_lon_errs, weekday_correct, hour_correct, n_points = [], [], [], [], 0
    _gen_t0 = _time.time()
    with torch.no_grad():
        for start in range(0, len(test_groups), batch_size):
            idx = list(range(start, min(start + batch_size, len(test_groups))))
            latlon, weekday_oh, hour_oh, mask, lengths = _pad_batch(
                test_groups, idx, max_traj_len, lat_c, lon_c, scale)
            latlon, weekday_oh, hour_oh, mask = (
                latlon.to(device), weekday_oh.to(device), hour_oh.to(device), mask.to(device))
            noise = torch.randn(len(idx), latent_dim, device=device)
            gen_latlon, gen_weekday, gen_hour = G(latlon, weekday_oh, hour_oh, noise, scale)

            gen_lat = (gen_latlon[..., 0] * scale + lat_c).cpu().numpy()
            gen_lon = (gen_latlon[..., 1] * scale + lon_c).cpu().numpy()
            gen_weekday_id = gen_weekday.argmax(dim=-1).cpu().numpy()
            gen_hour_id = gen_hour.argmax(dim=-1).cpu().numpy()
            mask_np = mask.cpu().numpy()

            real_lat = (latlon[..., 0] * scale + lat_c).cpu().numpy()
            real_lon = (latlon[..., 1] * scale + lon_c).cpu().numpy()
            real_weekday_id = weekday_oh.argmax(dim=-1).cpu().numpy()
            real_hour_id = hour_oh.argmax(dim=-1).cpu().numpy()

            for b, i in enumerate(idx):
                g = test_groups[i]
                L = g["length"]
                tail = slice(max_traj_len - L, max_traj_len)
                rows.append(pd.DataFrame(dict(
                    user_id=g["user_id"], traj_id=g["traj_id"], ts=g["ts"],
                    lat=gen_lat[b, tail], lon=gen_lon[b, tail],
                    weekday=gen_weekday_id[b, tail].astype(int),
                    hour=gen_hour_id[b, tail].astype(int),
                )))
                m = mask_np[b, tail].astype(bool)
                abs_lat_errs.append(np.abs(gen_lat[b, tail][m] - real_lat[b, tail][m]))
                abs_lon_errs.append(np.abs(gen_lon[b, tail][m] - real_lon[b, tail][m]))
                weekday_correct.append((gen_weekday_id[b, tail][m] == real_weekday_id[b, tail][m]))
                hour_correct.append((gen_hour_id[b, tail][m] == real_hour_id[b, tail][m]))
                n_points += int(m.sum())

    generation_time = _time.time() - _gen_t0
    synthetic_df = pd.concat(rows, ignore_index=True)
    synthetic_df = assign_grid_cells(synthetic_df, resolution_deg=0.01)
    synthetic_df = synthetic_df[["user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"]]
    fields = dict(
        n_points_generated=n_points,
        mean_abs_lat_error=float(np.concatenate(abs_lat_errs).mean()) if abs_lat_errs else float("nan"),
        mean_abs_lon_error=float(np.concatenate(abs_lon_errs).mean()) if abs_lon_errs else float("nan"),
        weekday_argmax_accuracy=float(np.concatenate(weekday_correct).mean()) if weekday_correct else float("nan"),
        hour_argmax_accuracy=float(np.concatenate(hour_correct).mean()) if hour_correct else float("nan"),
    )
    return synthetic_df, fields, generation_time


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def _cached_trajectory_chunks(df, max_traj_len, min_traj_len, cache_dir, cache_key, group_by="trajectory"):
    """Chunking (`_trajectory_chunks`) is a Python-level groupby loop over
    every group -- slow on a full-size dataset (minutes, paid twice:
    once for train, once for test) and identical every time run_trajgan
    is called again for the SAME dataset/windowing params, e.g. once per
    encoder/attacker seed. Caches the two chunk lists to disk (pickle)
    keyed by `cache_key` (caller's responsibility to make this unique
    per dataset + max_traj_len/min_traj_len/group_by combination) so the
    prep cost is paid once per cache_key, not once per call. Filename
    tagged "chunksv2" so this never collides with (or silently loads) a
    stale pre-chunking "trajgan_groups_*" cache file from before the
    2026-11-09 chunking fix."""
    import pathlib
    import pickle
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"trajgan_chunksv2_{cache_key}_{max_traj_len}_{min_traj_len}_{group_by}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    train_chunks = _trajectory_chunks(df, "train", max_traj_len, min_traj_len, group_by=group_by)
    test_chunks = _trajectory_chunks(df, "test", max_traj_len, min_traj_len=1, group_by=group_by)
    if cache_dir is not None:
        with open(cache_path, "wb") as f:
            pickle.dump((train_chunks, test_chunks), f)
    return train_chunks, test_chunks


def run_trajgan(df: pd.DataFrame, device: torch.device, max_traj_len: int = 150,
                 min_traj_len: int = 5, latent_dim: int = 100, lstm_units: int = 100,
                 batch_size: int = 64, max_epochs: int = 60, lr: float = 1e-3,
                 p_bce: float = 1.0, p_latlon: float = 10.0, p_weekday: float = 1.0,
                 p_hour: float = 1.0, seed: int = 0, cache_dir=None, cache_key: str = "",
                 group_by: str = "trajectory", checkpoint_every: int = 5):
    """Trains LSTM-TrajGAN on `df[df.split=="train"]`, then generates a
    synthetic counterpart of EVERY `df[df.split=="test"]` POINT (fresh
    noise draw per chunk) -- test trajectories longer than max_traj_len
    are split into consecutive chunks of at most max_traj_len points
    (`_trajectory_chunks`), never truncated, so no test point is ever
    dropped from the synthetic release (PI fix, 2026-11-09: truncation
    used to silently drop everything past the first max_traj_len points
    of a longer trajectory, breaking point-for-point alignment with the
    real test split for affected users). Returns `(synthetic_df, summary)`:

    `synthetic_df`: unified-schema dataframe (`puffin.data.schema.
    REQUIRED_COLUMNS`) of the synthetic trajectories -- same `user_id`/
    `traj_id`/`ts` as the real test trajectories they stand in for (so a
    TUL attacker's ground truth is unambiguous), synthetic `lat`/`lon`,
    `weekday`/`hour` (argmax of the generator's softmax heads), and
    `cell_id` re-tessellated from the synthetic (lat, lon) via
    `assign_grid_cells` (see module docstring for why).

    `summary`: n_train_trajectories, n_test_trajectories, d_loss_history,
    g_loss_history (per-epoch means, for an eyeball sanity check that
    training behaved -- D not collapsing to 0/1, G decreasing), and a
    utility proxy computed ONLY over true (non-padded) positions: mean
    absolute lat/lon error in the original's normalized units, and
    weekday/hour argmax accuracy against the real trajectory's own
    weekday/hour (how much of the real temporal signal survives
    generation).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    required = {"user_id", "traj_id", "ts", "lat", "lon", "weekday", "hour", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"run_trajgan: missing required columns: {missing}")

    train_groups, test_groups = _cached_trajectory_chunks(
        df, max_traj_len, min_traj_len, cache_dir, cache_key, group_by=group_by)
    if len(train_groups) == 0 or len(test_groups) == 0:
        raise ValueError("run_trajgan: need at least one usable trajectory in each split")

    lat_c, lon_c, scale = _latlon_centroid_scale(train_groups, test_groups)

    G = TrajGANGenerator(latent_dim, lstm_units).to(device)
    D = TrajGANDiscriminator(lstm_units).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    import copy
    import time as _time
    n_train = len(train_groups)
    d_loss_history, g_loss_history, epoch_times = [], [], []
    checkpoints = {}  # epoch (1-indexed) -> G.state_dict(), CPU tensors

    for epoch in range(max_epochs):
        _epoch_t0 = _time.time()
        order = rng.permutation(n_train)
        epoch_d, epoch_g, n_batches = 0.0, 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            latlon, weekday_oh, hour_oh, mask, lengths = _pad_batch(
                train_groups, idx, max_traj_len, lat_c, lon_c, scale)
            latlon, weekday_oh, hour_oh, mask = (
                latlon.to(device), weekday_oh.to(device), hour_oh.to(device), mask.to(device))
            lengths_t = torch.from_numpy(lengths).to(device)
            B = len(idx)

            # --- train D ---
            noise = torch.randn(B, latent_dim, device=device)
            with torch.no_grad():
                fake_latlon, fake_weekday, fake_hour = G(latlon, weekday_oh, hour_oh, noise, scale)
            d_real = D(latlon, weekday_oh, hour_oh)
            d_fake = D(fake_latlon, fake_weekday, fake_hour)
            d_loss = 0.5 * (F.binary_cross_entropy(d_real, torch.ones_like(d_real)) +
                             F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- train G ---
            noise = torch.randn(B, latent_dim, device=device)
            gen_latlon, gen_weekday, gen_hour = G(latlon, weekday_oh, hour_oh, noise, scale)
            d_pred_fake = D(gen_latlon, gen_weekday, gen_hour)
            g_loss = _traj_loss(mask, lengths_t, latlon, gen_latlon, weekday_oh, gen_weekday,
                                 hour_oh, gen_hour, d_pred_fake, p_bce, p_latlon, p_weekday, p_hour)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_d += float(d_loss.item())
            epoch_g += float(g_loss.item())
            n_batches += 1
        if n_batches > 0:
            d_loss_history.append(epoch_d / n_batches)
            g_loss_history.append(epoch_g / n_batches)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(_time.time() - _epoch_t0)
        actual_epoch_num = epoch + 1
        if checkpoint_every and actual_epoch_num % checkpoint_every == 0:
            checkpoints[actual_epoch_num] = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}

    # --- generate synthetic counterparts of every test trajectory, using
    # the FINAL-epoch G by default; the driver can call
    # generate_from_generator() again with a checkpoint-restored G
    # instead, if the collapse check below flags this run. ---
    synthetic_df, gen_fields, generation_time = generate_from_generator(
        G, test_groups, max_traj_len, latent_dim, lat_c, lon_c, scale, device, batch_size)

    summary = dict(
        n_train_chunks=len(train_groups),
        n_test_chunks=len(test_groups),
        d_loss_history=d_loss_history,
        g_loss_history=g_loss_history,
        checkpoints=checkpoints,
        regen_state=dict(test_groups=test_groups, max_traj_len=max_traj_len, latent_dim=latent_dim,
                          lstm_units=lstm_units, lat_c=lat_c, lon_c=lon_c, scale=scale),
        **gen_fields,
        final_d_loss=d_loss_history[-1] if d_loss_history else float("nan"),
        final_g_loss=g_loss_history[-1] if g_loss_history else float("nan"),
        epoch_times_seconds=epoch_times,
        mean_epoch_time_seconds=float(np.mean(epoch_times)) if epoch_times else float("nan"),
        generation_time_seconds=generation_time,
        lat_centroid=lat_c, lon_centroid=lon_c, scale_factor=scale,
    )
    return synthetic_df, summary
