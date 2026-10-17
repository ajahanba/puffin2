# Changes from the original Mo-PAE / PUFFIN code

Source of truth for "original": https://github.com/YutingZhan/Mo-PAE `mopae.py`
(cloned into `legacy/Mo-PAE_upstream/`), cross-checked against Utkarsh Darbari's
PUFFIN fork (`Mo-PAE_og_fairness/`, `Mo-PAE__fixed_naive/`, both on erebor under
`/home/NETID/ud21/`).

## Phase 0 (this port, `puffin/models/mopae_torch.py`, `puffin/data/geolife.py`)

Both `legacy` mode and `fixed` mode live side by side so the two can be compared
on identical data. `legacy` mode intentionally reproduces every bug below; it
is not meant to be used for any real result, only as the audit baseline.

| # | Issue (see `reports/00_mopae_audit.md`) | legacy mode | fixed mode |
|---|---|---|---|
| 1 | Eval only on train | reports train metrics only used for comparison, but splits train/test the buggy way | evaluates on genuinely held-out per-user test windows |
| 2 | Windows span users, labeled by first point | `legacy_windows()`: slides over the full concatenated location array, target id = window's first point | `fixed_windows()`: windows built within a single (user, trajectory) group; never crosses a user or trajectory boundary |
| 3 | Split after windowing | split happens on the already-windowed arrays, by position | each user's own trajectory is split 80/20 by time *before* windowing |
| 4 | Non-sequential input, ids as raw floats | `LegacyEncoder`: one LSTM timestep, `context` raw floats as "features" | `FixedEncoder`: real `context`-length sequence, `nn.Embedding` for location + weekday + hour |
| 5 | `from_logits=True` on a softmax output (double softmax) | reproduced exactly via `_double_softmax_xent` | single `F.cross_entropy` on raw logits (no final softmax layer) |
| 6 | Reconstruction target is a meaningless location-ID magnitude | decoder reconstructs the raw location id (unchanged) | decoder reconstructs (lat, lon) grid coordinates |
| 7 | λ signs baked in, one shared optimizer | `LEGACY_LAMBDAS` keeps the original signed convention, one optimizer over all four sub-networks | `FIXED_LAMBDAS` are all positive; the adversarial minus sign is explicit in the encoder's loss expression; four independent optimizers |
| 8 | Privacy measured only by the in-model adversary | unchanged — out of scope for Phase 0, this is Phase 3's external-attacker suite | same — `internal_reid` in the audit output is still the in-model adversary; do not read it as a privacy guarantee |

Note on issue #5: Utkarsh's PUFFIN fork (`fairmopae.py` in both of his
directories) already uses `from_logits=False` — this specific bug did **not**
carry over into PUFFIN. `legacy` mode here still reproduces it because it is
meant to match published upstream `mopae.py`, not the PUFFIN fork.

Everything else from PUFFIN's fork (issues #2, #3, #4, #6, #8, and the shared
optimizer in #7) is present unchanged in both of Utkarsh's directories, so any
number from the rejected PUFFIN draft inherits them.

## Fairness losses (`puffin/fairness/losses.py`)

Ported from `models_mopae.py`'s `fairness=` branches, per the PI's decision
after Checkpoint 0.

- **`ed` → `ed_exact` + `ed_soft`.** `ed_exact` is a faithful port of
  `fairness="ed"`. **New finding, not in the Phase 0 audit:** the original
  `ed` loss computes the *predicted* user via `tf.argmax(y3_pred)` then
  `tf.one_hot(...)`, both gradient-free in TensorFlow. Every term in that
  loss is therefore a constant with respect to every trainable variable —
  `fairness="ed"` logged a number but contributed **zero training signal**.
  This is the *default* `fairness=` value in `Mo-PAE_og_fairness/fairmopae.py`
  — i.e. the config most likely to have produced the rejected PUFFIN draft's
  reported fairness numbers. Verified in PyTorch: `ed_exact(...).requires_grad
  == False`; `scripts/smoke_test_fairness.py` confirms this directly. This is
  an additional, independent reason (beyond issues #2–#4, #6, #8) the old
  draft's numbers shouldn't be presented, even as a pilot.
  `ed_soft` replaces the hard argmax with the predicted *distribution*
  (`probs @ entropy_table`, an expectation instead of a point estimate), so
  it has a real gradient and is usable. Wired into `fixed_train_step` as the
  Phase 4 **L_pair_threshold baseline**, off by default (`lambda4=0.0`,
  `fairness_fn=None`) so Phase 0's already-reported numbers stay reproducible.
- **`confidence` → `confidence_exact`.** Faithful port (reduce_max → `.max()`,
  the TF code's `custom_loss` threshold gate reproduced exactly, including
  that every statistic — true entropy, predicted entropy, confidence — is a
  single scalar over the *whole batch*, not per-sample; and the whole term
  returns exactly 0 whenever that batch's max entropy-gap already clears
  0.1). Reference/logging only, not wired into training — ported because it
  took under an hour, as instructed, not because it's recommended for use.
- **`matches`, `naive` — reference only, not ported.** Their original TF
  formulas (for anyone who wants them later):
  - `matches`: `loss4 = reduce_sum(onehot(argmax(y3_pred)) * y3_true * y4)` —
    also routes through `tf.argmax`/`tf.one_hot`, so it is inert for the same
    reason as `ed_exact`.
  - `naive`: `loss4 = reduce_std(y3_pred * y4)` — standard deviation of the
    predicted-distribution-weighted entropy vector; this one *is*
    differentiable (no argmax), unlike `ed`/`matches`.
  Source: `models_mopae.py` in both `/home/NETID/ud21/Mo-PAE_og_fairness/`
  and `/home/NETID/ud21/Mo-PAE__fixed_naive/` on erebor.

## Infrastructure

- PyTorch reinstalled with a CUDA 12.1 wheel (`torch==2.5.1+cu121`) matching
  the erebor driver (535.54.03, max CUDA 12.2) — the first install resolved a
  CPU-only CUDA-13 build. All four A4500s confirmed visible
  (`torch.cuda.device_count() == 4`). Done entirely inside `.venv`; the system
  driver was not touched.
- Code (not data, not results) pushed to a private GitHub remote:
  `https://github.com/ajahanba/puffin2` (private, via the `gh` CLI already
  authenticated on erebor as `ajahanba`).
- `/data` is included in erebor's system-level backups — confirmed by seeing
  `mashhadi`'s files under `/backup/dataBackup/` and a more recent
  `/backup/dataBackup09172026/` snapshot. Snapshot cadence/retention policy
  is not visible from a regular user account; ask erebor's sysadmin if that
  matters for anything irreplaceable.
- Environment lockfile: `requirements.lock.txt` (`pip freeze` inside `.venv`),
  generated after an unexpected erebor reboot mid-session corrupted two
  packages (matplotlib, contourpy — both left as 0-byte `__init__.py`,
  caught mid-write by the crash) so the exact working set is pinned and
  reproducible going forward.

## Phase 1 bugs found and fixed (data pipeline)

- **Index-misalignment bug** (`puffin/data/datasets/_checkin_utils.py`):
  `assign_daily_traj_id` returned a pandas Series carrying its own fresh
  0..n-1 index; when the caller's dataframe had a non-contiguous index
  (Gowalla's US bounding-box filter drops rows), the `df[...] = ...`
  assignment silently misaligned by label instead of position, NaN-filling
  ~32% of Gowalla's rows. Foursquare's loader shared the same code path but
  never filters its input first, so it never manifested there. Fixed to
  return a plain array (always positional); added a regression test
  (`tests/test_checkin_utils.py`) with a deliberately non-contiguous index.
- **Quadratic bug** (`puffin/data/splits.py`): `membership_split` recomputed
  a full-dataframe boolean mask (`df["user_id"] == u`) once per *member*
  user — O(n_users × n_rows). Fine up to tens of thousands of users; on
  YJMob100K (100,000 users × 111.5M rows) it did not finish in over 90
  minutes and had to be killed (found via a `py-spy` stack trace on the
  stuck process). Rewrote `membership_split` and `temporal_split`
  (same O(n_users) Python-loop class of issue) to use vectorized
  groupby/merge operations; `stats.per_user_entropy`'s `.apply(lambda...)`
  fixed the same way. Re-run: 4m09s (vs. killed after 90+ min). Verified via
  `tests/test_splits_stats.py` and a byte-identical Gowalla-NYC rebuild
  pre/post-fix.

## Phase 2 — PI-directed correction after Checkpoint 2

**Check-in windowing (Foursquare, Gowalla): dropped the daily-trajectory
requirement.** The Checkpoint 2 version built SL=10 windows within each
`traj_id` group, and for Foursquare/Gowalla that `traj_id` is not a real
trajectory — `_checkin_utils.assign_daily_traj_id` invents one calendar-day
segment per user, purely because those datasets have no native trajectory
id. Requiring 10 check-ins inside a single calendar day left only 71/1083
Foursquare users with even one usable window, and that 71 is not a random
subset — it's whoever happens to check in unusually often. Per the PI:
`puffin/attacks/windowing.py`'s `build_sl_windows` now takes
`respect_traj_boundary` (default `True`, used for GeoLife's real GPS
trajectories, unchanged) — `False` builds windows over each user's full
chronological check-in sequence instead, still never crossing the user or
train/test-split boundary. Effect, confirmed in `reports/02b_disparate_risk_v2.md`:
Foursquare went from 71/1083 to **1083/1083** users with ≥1 test window at
SL=10; Gowalla-NYC (not run at Checkpoint 2) reaches 587/1517 at SL=10 and
1154/1517 at SL=5. This also measurably changed the finding, not just the
sample size — see that report for why.

Also hardened per the PI: the raw attacker is now "best of MLP vs. GRU" per
seed (chosen by validation accuracy, 5 seeds, early stopping) instead of a
single architecture trained for a fixed small number of epochs; the risk
regression now reports proper standard errors/p-values
(`puffin/attacks/analysis.py`) instead of point estimates only; decile means
carry bootstrap 95% CIs over users.
