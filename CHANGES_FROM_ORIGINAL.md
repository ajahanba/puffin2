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

## Phase 3 task 3 — membership inference: POIPrivacy tried and ruled out, in-house fallback built

Per the plan's explicit time-box ("if POIPrivacy doesn't run on our data
after ~half a day of real effort, stop, implement an in-house equivalent,
document why, and move on" — `PROGRESS.md`).

**POIPrivacy was cloned and given a genuine, bounded attempt**
(`legacy/POIPrivacy/`, not committed, same convention as
`legacy/Mo-PAE_upstream/`; `.gitignore` updated the same way): its README
and all four attacks' code (LocExtract, TrajExtract, LocMIA, TrajMIA) were
read in full, a fresh isolated venv was built
(`legacy/POIPrivacy/.venv`, this project's `.venv` untouched), its
non-torch dependencies installed cleanly, and `torch==2.5.1+cu121` (this
project's own pinned wheel) installed in place of its pinned
`torch==1.12.1+cu113` without issue. Its own demo was then actually run:
`Attack3_LocMIA/build_graph.py` against the bundled 4sq-NYC sample data
executed successfully end to end. **POIPrivacy is not abandoned or broken
code** — it was ruled out for a structural reason, found by reading what
that successful demo run actually produced and what the attack code
actually needs:

1. **LocMIA and TrajMIA (the two attacks that correspond to "is this
   user's data in the training set") are both built around GETNext**, a
   specific graph-convolution + transformer POI *recommender*
   (`Attack3_LocMIA/model_batch.py`, `Attack4_TrajMIA/model.py`), not a
   generic "any next-location model" interface. There is no supported path
   to point either attack at a different target architecture — the
   recommender *is* the attack's target model, hardcoded throughout
   `train_target.py`/`traintarget.py`.
2. **GETNext's graph is a dense POI-to-POI adjacency matrix written as a
   single `O(n_pois^2)` CSV** (`build_graph.py`'s `save_graph_to_csv`:
   `np.savetxt(..., A.todense(), delimiter=',')`). Running it against the
   bundled 4sq-NYC sample (this attempt's own test run) produced a 519MB
   dense CSV for roughly 11,000 POIs. GeoLife — this project's main
   dataset — has 85,074 distinct grid cells
   (`data/processed/geolife/stats.json`); a dense 85074×85074 float32
   matrix alone is ~29GB, before a single GCN forward pass is run on it.
   This is not a performance-tuning problem, it rules GeoLife out outright.
3. **GETNext's node features and an auxiliary training loss are built on
   POI categories** (`poi_catid`/`poi_catname`, one-hot encoded) **that do
   not exist anywhere in this project's data.** `puffin/data/schema.py`'s
   unified schema — what every processed parquet in this repo conforms to
   — has no category field: meaningless for GeoLife's raw GPS fixes (a GPS
   point isn't a venue with a type), and dropped even for Foursquare's
   check-ins by `puffin/data/datasets/foursquare_unified.py`, since the
   unified schema has nowhere to put it. Recovering it would mean
   fabricating label data this project's pipeline was deliberately built
   not to depend on.
4. On top of both, LocMIA/TrajMIA are LiRA-style (shadow-model) attacks —
   `attack3_shadowmodel.py`/`train_shadow.py` train many independent
   copies of this same heavy GCN+transformer model, multiplying problem
   #2's memory wall by however many shadow models the attack needs.

Items #2 and #3 are each, independently, exactly the stop condition the
plan names ("undocumented/unrecoverable data format assumptions") — not a
dependency-version problem a pin would fix, but a structural mismatch
between what GETNext's MIA attacks need (a modest-cardinality, category-
labeled POI graph) and what this project's data is (large-vocabulary GPS
grid cells with no category labels at all).

**In-house fallback: `puffin/attacks/mia_inhouse.py`.** A classic Yeom et
al. (2018) / Shokri et al. loss-threshold MIA — deliberately the simplest
attack in the MIA literature, not a LiRA/shadow-model attack, since this
module's job is to demonstrate the standard member/nonmember loss
asymmetry exists on our data and give later phases a real per-user signal,
not to find the strongest possible attacker (if that's ever needed,
POIPrivacy's shadow-model machinery is where that escalation would return
to scope). Ground truth comes directly from `mi_split`
(`puffin.data.splits.membership_split`, already in every processed
parquet). The attacked target model is Mo-PAE-fixed's own
`FixedEncoder` + `MLPHead` (`puffin/models/mopae_torch.py`) — i.e. "the
utility (next-location) model trained on released features `f`" that
`puffin/attacks/README.md`'s threat-model table names as Mo-PAE/PUFFIN's
MIA attack surface — trained directly on raw member windows via plain
cross-entropy (no adversarial/privacy terms: those are Mo-PAE's own
defense mechanism, `fixed_train_step`, orthogonal to needing *a* trained
next-location model to attack).

`run_loss_threshold_mia(df, device, ...)` builds membership-keyed windows
by reusing `puffin.attacks.windowing.build_sl_windows` unchanged (relabels
`mi_split` as the `split` column it already expects), trains the target
model on member windows, then calibrates a single confidence threshold on
a random half of each class's windows (member and nonmember, independently
sampled) and reports attack performance only on the other half — so the
threshold was never set by looking at the instances it later classifies.
Returns a per-user DataFrame (`user_id`, member/nonmember window counts,
mean confidence on each, `attack_accuracy`) plus a summary dict
(calibrated threshold, overall AUC, overall accuracy) — per-user, not just
an aggregate number, matching this project's whole per-user-risk
methodology (`puffin/attacks/reid.py`'s convention).

Tested two ways: `tests/test_mia_inhouse.py` (run via
`python tests/test_mia_inhouse.py`) checks `_windows_by_membership` never
mixes member/nonmember windows (including a deliberately-asymmetric-length
case that would catch a leak), `_best_threshold`'s ROC-based threshold
selection against a clean synthetic signal and a degenerate single-class
calibration split, and an end-to-end run on synthetic data confirming the
textbook overfitting asymmetry (member windows end up more confidently
predicted than never-seen nonmember windows) — all five checks pass. Also
run once against the real GeoLife parquet (`data/processed/geolife/`,
181 users, SL=10, capped at 50 windows/user, 12 epochs, one A4500, 73.7s
wall time): overall AUC 0.555, overall accuracy 0.598 over 4,056
member / 4,128 nonmember eval windows — a real, modest-but-above-chance
signal (not suspiciously perfect, consistent with a model that's seen only
12 epochs over GeoLife's 85,074-location vocabulary rather than having
memorized it), with per-user `attack_accuracy` ranging from near 0 to 1
across the 181 users. That run was a validation step only, not committed
as a script or a results artifact — Phase 3 task 3's deliverable is the
module and its test, not a full sweep.

## Phase 3 task 4 — trajectory-user linking: MainTUL ported (not ruled out)

Per the plan's time-box (PROGRESS.md): give the named third-party repo a
genuine, bounded attempt (~half a day) before falling back to an in-house
equivalent. Unlike task 3's POIPrivacy, **MainTUL actually ran**, so the
outcome here is a port of its real method, not an in-house TULER fallback.

**MainTUL was cloned and given a genuine attempt**
(`legacy/MainTUL/`, not committed, same convention as
`legacy/Mo-PAE_upstream/` and `legacy/POIPrivacy/`; `.gitignore` updated
the same way): its README and all of `project/{preprocess,loaddataset,
model,main,utils}.py` were read in full, a fresh isolated venv was built
(`legacy/MainTUL/.venv`, this project's own `.venv` untouched), and
`torch==2.5.1+cu121` (this project's own pinned wheel) installed in place
of its pinned `torch==1.10.1+cu102` without issue (plus `scikit-learn`,
`pandas`, `tqdm`, `matplotlib`). Its own `foursquare_mini` demo was then
run (`python project/main.py --dataset foursquare_mini --epochs 2
--batch_size 64`, from the repo root as its own logging path requires)
and hit two bugs before training, both plain version/code bugs, neither a
data-format or architecture problem:

1. `utils.py`'s `EarlyStopping.__init__` sets `self.val_loss_min = np.Inf`
   -- removed in numpy>=2.0 (`AttributeError`); `np.inf` is identical and
   still works. One-line fix.
2. `main.py`'s per-batch training-loop log line formats
   `"...acc@1:{:.6f} acc@5:{:.6f} macro_p:{:.6f} macro_r:{:.6f}
   macro_f1:{:.6f}"` (5 more placeholders beyond loss) against only the 7
   loss-related values the TRAINING loop actually has -- those 5 metrics
   are only ever computed in the VALIDATION loop a few lines later.
   `str.format` raises `IndexError` on the very first logged batch
   (`batch_idx % 40 == 0` is true at batch 0) under the authors' own
   default args, independent of numpy/torch version -- this looks like
   the published repo's own default training config was never actually
   run end-to-end in its current committed state. Trimmed the format
   string to the 7 metrics the training loop has.

With both fixed, the demo trained and tested successfully end to end: 2
epochs, batch_size 64, ~200s single-threaded CPU, logged test performance
`acc@1 0.573 / acc@5 0.684` (student) and `acc@1 0.499 / acc@5 0.613`
(teacher). **This is a real, working demo run on the authors' own data**
-- a materially different outcome from task 3's POIPrivacy, which also ran
its own demo but was then ruled out for structural reasons found by
reading what that demo actually needed (GETNext's O(n_pois^2) graph and
POI-category features). MainTUL's architecture (read directly from
`model.py`, not guessed from the README) is a mutual-distillation pair --
a lightweight LSTM "student" and a Transformer "teacher", each with its
own classifier head and a bidirectional KL-divergence term distilling
between their output distributions, trained so the cheap student alone
becomes as accurate as the Transformer teacher and can be deployed on its
own at inference. Its only architectural dependency that this project's
schema lacks is a POI category id, fused in via a learnable alpha-weighted
sum alongside the POI-id embedding in both encoders -- unlike POIPrivacy's
GETNext graph, this is NOT a hard blocker (no graph construction, no
quadratic memory wall): the category branch can simply be omitted.

## The port: `puffin/attacks/tul_maintul.py`

Because MainTUL ran, it was PORTED (not imported -- `legacy/MainTUL/` is
not committed, so nothing under `puffin/` may depend on it at runtime)
into this project's own schema and conventions, rather than falling back
to a simpler in-house TULER baseline. What was ported as-is vs. simplified,
and why (full reasoning also in the module's own docstring):

- **Category view dropped entirely**, for the reason above: GeoLife (this
  validation's dataset) has no POI category info at all -- a GPS fix isn't
  a venue with a type, the same reasoning this file's POIPrivacy section
  already gave for why `puffin/data/schema.py` has nowhere to put one.
  Faking a constant dummy category would run, but would only add dead
  parameters for zero signal while looking like real data; the port keeps
  the single POI-id view instead.
- **Trajectory length capped** at `max_traj_len` (default 150). GeoLife's
  real per-trip GPS trajectories run up to 92,645 points
  (`df.groupby(["user_id","traj_id"]).size().max()` on
  `data/processed/geolife/unified.parquet`) -- orders of magnitude longer
  than the check-in trajectories (tens of points/day) MainTUL was built
  for; a Transformer encoder over a 92,645-token sequence is quadratic-
  attention infeasible. Each trajectory is truncated to its first
  `max_traj_len` points (chronological order) -- a practical necessity,
  not a claim that longer context wouldn't help.
- **Train/test split reuses this project's own `mi_split` column**
  (`puffin.data.splits.membership_split`: whole trajectories held out per
  user, never split mid-sequence) instead of re-deriving MainTUL's own ad
  hoc 80/20 `split_dataset` -- same default fraction
  (`nonmember_traj_frac=0.2`) and the same "whole trajectory" property,
  reusing existing infra instead of duplicating it (the same pattern
  `mia_inhouse.py`'s `_windows_by_membership` already established).
  `user_disjoint` users are EXCLUDED here, unlike `mia_inhouse.py`: TUL is
  a CLOSED-WORLD classification problem (the classifier's output layer has
  one slot per training user), so a disjoint user's trajectory is
  guaranteed-wrong by construction, not a real attack signal -- MIA's
  open-world threshold test has no such constraint, which is why
  `mia_inhouse.py` keeps them.
- **Long-term augmentation kept** -- MainTUL's actual "mutual
  distillation" trick (during training only, the teacher sees the
  student's trajectory concatenated with up to `long_term_num` other
  random trajectories of the same user, for richer soft targets; at test
  time both networks see only the single trajectory being attacked, no
  side information). The original crashes (`random.sample` on a
  population smaller than `long_term_num`) for any user with too few
  training trajectories; `long_term_num` is clamped per user here instead
  (46/181 GeoLife users have fewer than 9 train trajectories after the
  80/20-equivalent split).
- **Temporal encoding uses a within-sequence position index**, not a raw
  unix timestamp (this project's `ts` is a `datetime64`, and the original
  only ever used the raw timestamp as an arbitrary large number fed
  through a learnable per-dimension frequency anyway).
- **Not ported**: `--times` multi-seed repeats, file logging, loss-curve
  plotting, multi-GPU `DataParallel`, MainTUL's own file-based mid-training
  checkpoint -- experiment-harness scaffolding, not the method. Uses the
  in-memory best-state pattern (`copy.deepcopy` + early stopping on
  validation accuracy) `puffin.attacks.reid` already uses, and
  `reduction="batchmean"` in place of `F.kl_div`'s deprecated
  `size_average=False` (identical computation, no deprecation warning).

**Why this is structurally different from `puffin.attacks.reid.py`, not
the same thing relabeled**: `reid.py`'s GRU/MLP attackers always consume a
FIXED-length SL window (sliding windows carved out of a trajectory,
trajectory identity discarded) -- never padded, never masked, because
every input is exactly `sl` long. TUL's actual object is an entire
VARIABLE-length released trajectory, so `tul_maintul.py` needs real
sequence machinery `reid.py` never does (padding, a Transformer key-
padding mask, `pack_padded_sequence` for the LSTM), and the long-term
augmentation trick has no analogue for a fixed-length window (there's no
notion of "a longer version of this window" the way there is for "this
user's other trajectories"). The distinction named in the task
instructions is real, not cosmetic.

**Output convention**: `run_tul_maintul(df, device, ...)` returns
`(per_user_df, summary)`, matching `reid.py`/`mia_inhouse.py`'s
established per-user-not-just-aggregate convention. `per_user_df` has one
row per user with >=1 train trajectory: `user_id`, `n_train_trajectories`,
`n_test_trajectories`, `student_top1`/`top5`, `teacher_top1`/`top5`, and
both networks' mean confidence on their test trajectories' true label.

**Tested**: `tests/test_tul_maintul.py` (run via
`python tests/test_tul_maintul.py`) checks `build_trajectory_dataset`
excludes `user_disjoint` users entirely (closed-world requirement),
correctly relabels `mi_split` into train/test and truncates to
`max_traj_len`, that `_sample_long_term` clamps instead of crashing for
users with few trajectories, and an end-to-end run on synthetic data with
user-specific cell-id bands confirming both the student and teacher link
held-out trajectories well above chance -- all four checks pass.

**Real validation run on GeoLife**
(`data/processed/geolife/unified.parquet`, one A4500, `embed_size=128,
hidden_size=256, num_layers=2, num_heads=4, long_term_num=8,
max_traj_len=150, batch_size=128`, early-stopped within 25 max epochs,
~17.6 minutes wall time): 163 non-disjoint users, 13,611 train trajectories
(12,196 fit / 1,415 validation) and 3,403 test trajectories. **Overall
(trajectory-pooled) test performance: student top-1 0.557, top-5 0.795;
teacher top-1 0.569, top-5 0.803** -- chance is ~1/163 ≈ 0.006, so this is
a large, real signal, and the student (the network actually meant for
deployment after distillation) is essentially as good as the teacher,
which is the whole point of the method. The per-user distribution is
notably skewed, though, in a way worth flagging rather than hiding behind
the pooled number: median per-user `student_top1` is only 0.071 (153/163
users have >=1 test trajectory; 25th percentile is exactly 0), while a
handful of very-prolific users (e.g. one user with 1,722 train / 431 test
trajectories) get linked at 0.79-0.85 top-1 alone -- the pooled metric,
being trajectory-weighted, is dominated by these heavy users. This mirrors
the disparate-risk pattern Checkpoints 2/2b/2c already found for re-id
risk (more data from a user = more risk to that user), now observed for
TUL too; worth carrying into the real attack once a TrajGAN/GI-DP
baseline exists to attack for real, rather than re-discovering it then.
This run used `scripts/smoke_test_phase3_tul.py` (committed, same
convention as task 2's `scripts/smoke_test_phase3_attacker.py`), not a
results artifact -- Phase 3 task 4's deliverable is the module and its
test, not a full sweep.

## Phase 5 task 1 — TrajGAN: LSTM-TrajGAN ported (ran out of the box, one trivial fix)

Per the plan's time-box (PROGRESS.md): give the named third-party repo a
genuine, bounded (~half a day) attempt before falling back to an in-house
equivalent. **LSTM-TrajGAN's own demo ran essentially out of the box** --
a cleaner outcome than either task 3's POIPrivacy (structural blockers) or
task 4's MainTUL (two bugs to fix before it ran).

"TrajGAN" was confirmed to mean Rao, Gao, Kang & Huang's **LSTM-TrajGAN**
(GIScience 2021, arXiv:2006.10521) before touching anything -- the paper's
abstract, architecture, and TUL-based evaluation protocol all match the
expected "well-known trajectory-generation GAN with privacy framing."
Official code: https://github.com/GeoDS/LSTM-TrajGAN, cloned to
`legacy/LSTM-TrajGAN/` (not committed, same convention as every other
third-party clone in this project; `.gitignore` updated the same way).

**The attempt**: a fresh, isolated Python 3.6.8 venv
(`legacy/LSTM-TrajGAN/.venv`, this project's own `.venv` untouched; erebor
has a system `python3.6` matching the repo's own pinned "Python 3.6.3")
with its exact pinned requirements (numpy==1.18.4, pandas==1.1.5,
tensorflow==1.13.1 -- the CPU build, not tensorflow-gpu: erebor's driver
(535.54.03) only supports up to CUDA 12.2, while TF 1.13 needs CUDA 10.0,
so GPU support for this 2019-era TF build isn't realistically installable
here; not a real loss since the repo's own demo is small and fast on CPU
anyway, and training entirely on CPU also means this baseline never
competes with the Phase 4 grid's GPU usage -- Keras==2.2.4, geohash2==1.1,
scikit-learn==0.23.2). All installed cleanly on the first try.

`python train.py 20 256 10` (20 epochs, the repo's own bundled Foursquare-
NYC-derived demo data in `data/final_train.npy`) **ran to completion with
zero code changes**: D loss fell 32.3 -> 8.2, G loss fell 2094 -> 1986 over
20 epochs -- a sane, non-collapsing adversarial training curve, logged by
the script itself. The repo also ships pretrained checkpoints
(`params/G_model_2000.h5` and all of `training_params/{G,D,C}_model_*.h5`
for every 10th-100th epoch up to 2000) and a pre-existing
`results/syn_traj_test.csv` from the original authors' own run --
independent evidence this repo was genuinely functional before this
project touched it.

**One real bug, found at the generation step** (`predict.py`, not
training): `keras.engine.saving.load_weights_from_hdf5_group` calls
`f.attrs['keras_version'].decode('utf8')`, assuming h5py's pre-3.0
bytes-valued attrs API. `pip install Keras==2.2.4` resolves an unpinned,
current `h5py` (3.1.0 as of this attempt) whose attrs are already `str` --
`AttributeError: 'str' object has no attribute 'decode'` on the very first
`load_weights` call. Fixed by pinning `h5py==2.10.0` (contemporaneous with
Keras 2.2.4/TF 1.13.1). After that one-line dependency pin, `predict.py 20`
ran cleanly and produced a real `results/syn_traj_test.csv` (22,153
synthetic rows) from the just-trained 20-epoch checkpoint. Same category
of bug as MainTUL's `np.Inf`/logging-format issues (task 4) -- a plain
dependency-version mismatch against 2019-era pinned requirements, not a
data-format or architecture problem.

### The port: `puffin/baselines/trajgan.py`

Because the original ran cleanly, it was PORTED (not imported -- the clone
isn't committed) into this project's PyTorch/schema conventions, reading
directly from the original's `model.py`/`losses.py`, same policy as
`tul_maintul.py` for MainTUL. Full architecture-level reasoning is in the
module's own docstring; summary of what changed and why:

- **`category` dropped entirely** -- the original's 10-class venue-category
  one-hot field has no equivalent anywhere in `puffin/data/schema.py`
  (same reasoning as `tul_maintul.py`'s POI-category drop, task 4, and
  the POIPrivacy writeup, task 3: a GPS fix isn't a venue with a type).
  Removes one embedding branch, one output head, one TrajLoss term.
- **Synthetic `cell_id` added via re-tessellation, not a GAN output.** The
  original only ever outputs continuous (lat, lon); but `tul_maintul.py`
  (the TUL attack this mechanism's output is meant to feed, per
  `puffin/attacks/README.md`'s threat table) consumes `cell_id`. After
  generation, `puffin.data.tessellation.assign_grid_cells` (the same
  function Phase 1 used to build every real dataset's `cell_id`) is run
  on the synthetic (lat, lon) to produce a dense, schema-valid `cell_id`.
- **Trajectories truncated to `max_traj_len`** (default 150, same default
  and same reason as `tul_maintul.py`): GeoLife's real trajectories run up
  to 92,645 points (`data/processed/geolife/unified.parquet`), far past
  the original's own `max_length=144` (sized for check-in data) and
  infeasible for an LSTM to train over directly.
- **Reuses this project's own `split` column** (`temporal_split`) for
  train/generate, instead of the original's own ad hoc split -- same
  "reuse existing infra" pattern as `tul_maintul.py`/`mia_inhouse.py`.
- **Training loop uses the original's own "regular" alternative** (full
  pass over all batches per epoch) rather than its *default* loop (one
  random batch per "epoch," repeated many times) -- the original's own
  code comment says the alternative "usually takes fewer epochs to
  converge," and it's the better choice for a validation run on real data
  rather than reproducing the original's exact epoch-counting quirk.
- **`scale_factor` uses the 99.5th percentile of absolute deviation from
  centroid, not the original's true max.** This was found, not assumed,
  while validating on real GeoLife: `data/processed/geolife/
  unified.parquet` contains a small number of physically-impossible GPS
  fixes (lat up to ~400 degrees, lon spanning the full -180..180 range --
  a known raw-GPS-noise characteristic of GeoLife, not introduced by this
  port or by Phase 1's pipeline). The original's `scale_factor = max(abs
  deviation)` would let one such glitch dominate the entire normalization,
  wasting the tanh output's dynamic range and making the reported
  mean-absolute-lat/lon-error utility metric report in units dominated by
  that one outlier. Normalized inputs are also clipped to [-1, 1] so an
  outlier point saturates instead of distorting the batch. Same class of
  judgment call as `tul_maintul.py`'s length truncation: a necessary,
  documented adaptation to make the published mechanism usable on this
  project's real (noisier) data, not a correctness fix to the original
  (whose own NYC check-in data apparently never needed it).
- **Not ported**: `geohash2`-based CSV encoding (irrelevant here --
  tessellation into `cell_id` replaces geohashing), the bundled
  MARC-based `TUL_test.py` evaluation (this project's own TUL attack,
  `tul_maintul.py`, is what the threat-model table names for this
  mechanism; running the original's bespoke classifier too would be a
  redundant second TUL implementation), per-epoch file checkpointing
  (kept in-memory only -- GANs have no clean "best epoch" criterion the
  way `tul_maintul.py`/`reid.py`'s validation-accuracy early stopping
  does).

**Tested**: `tests/test_trajgan.py` (3 checks) -- `_trajectory_groups`
correctly splits by the `split` column and truncates to `max_traj_len`;
`_pad_batch` pre-pads with exact zeros and masks real positions correctly
(including that each real timestep's one-hot weekday/hour sums to exactly
1); and an end-to-end run on synthetic data with user-specific lat/lon
patches confirms the trained generator's lat/lon reconstruction error is
less than half an untrained (0-epoch) generator's error on the same
data -- a real, measurable "did training work" signal, not just "did it
run." All three pass.

**Validated on real GeoLife** (`data/processed/geolife/unified.parquet`,
24,876,961 rows / 181 users, CPU, `max_traj_len=100, batch_size=128,
max_epochs=30`, via `scripts/smoke_test_phase5_trajgan.py`, 19m14s wall
time): 14,788 train / 3,941 test trajectories (370,179 synthetic points
generated). G loss fell from 24,875 (epoch 1) to 678 (epoch 30) -- a ~37x
reduction, real reconstruction learning. D loss fluctuated in a low range
(0.0003-0.068 across epochs, final 0.0022) rather than monotonically
decreasing -- the discriminator stayed roughly able to tell real from fake
even as G's reconstruction quality improved, an expected unregularized-GAN
dynamic given `p_latlon=10` dominates the loss (same weighting as the
original), not a sign of broken training (see the training-dynamics note
below). Mean absolute lat/lon reconstruction error on held-out test
trajectories fell from 148.6/144.8 degrees after 2 epochs to 9.7/16.5
degrees after 30 -- a 15x/8.8x improvement, though still coarse in
absolute terms within this half-day
time-box (GeoLife's raw-GPS outliers push `scale_factor` to 232 -- see the
robust-scale fix above -- so these aren't directly comparable to
`geolife_sample`'s much smaller-scale errors below). Weekday/hour argmax
accuracy: 0.185/0.090 vs. chance 0.143/0.042 -- real but modest, same
pattern as the `geolife_sample` run.

A faster validation pass on `geolife_sample` (136 users, 1,407 train /
421 test trajectories, 60 epochs, ~3.5 min wall time) showed the same
qualitative pattern at smaller scale: mean absolute lat/lon reconstruction
error on held-out test trajectories fell from 5.42/5.97 degrees after 2
epochs to 0.75/0.85 degrees after 60 -- a real, substantial improvement
from training, though still coarse in absolute terms (expected: `p_latlon`
dominates the loss by design, same as the original, so spatial
reconstruction improves fastest; weekday/hour argmax accuracy stayed
weak, 0.33/0.09 vs. chance 0.14/0.04 -- a real but modest signal,
consistent with those fields' much smaller loss weight, `p_weekday=p_hour=1`
vs. `p_latlon=10`, same as the original's own weighting).

**One training-dynamics observation worth flagging, not a bug**: the
discriminator's loss collapses toward ~0 well before training ends on
both the sample and real data (e.g. D loss 0.72 -> 0.0003 over 60 epochs
on `geolife_sample`) -- a textbook unregularized-GAN dynamic (D becoming
too strong relative to G, no gradient penalty or label smoothing in
either the original or this port), not something unique to this port; the
original's own 20-epoch demo run (D loss 32.3 -> 8.2, still decreasing,
not yet collapsed) simply didn't train long enough to show it. Worth
knowing before reading too much into G's adversarial loss term alone in
any future quantitative comparison -- the reconstruction terms
(`p_latlon`/`p_weekday`/`p_hour`) are doing most of the real work once D
saturates.

## Phase 5 task 2 — GI-DP: no canonical repo found; in-house Planar Laplace mechanism built

Unlike TrajGAN, **"GI-DP" is not a specific paper/repo name** -- read here,
per the task's own framing, as shorthand for **Geo-Indistinguishability**
(Andres, Bordenabe, Chatzikokolakis & Palamidessi, CCS 2013,
arXiv:1212.1984), the classical DP-flavored location-perturbation
mechanism. A genuine search was made for an existing implementation before
deciding to build in-house (not skipped in favor of writing code
immediately):

- **GeoPrivacy** (PyPI) implements the Planar Laplace noise sampler, but
  its own README scopes it to the **sporadic case** -- independent single
  location reports (its own example is one NYC taxi pickup), with no
  trajectory or dataframe integration at all.
- **privkit** (`privkit.fc.up.pt`) implements an *Adaptive*
  Geo-Indistinguishability variant (Al-Dhubhani & Cazalas 2018) that
  dynamically adjusts epsilon based on correlation with past locations --
  a materially different, more elaborate mechanism than "the classical
  mechanism" named in the task, and pulling in a whole external
  privacy-metrics framework for one formula would be a worse trade than
  writing the actual math.
- No GitHub repository literally named "GI-DP" implementing a
  trajectory-level mechanism was found via web search.

Given no fitting repo exists (not "a repo exists but is ruled out" --
genuinely nothing to clone and attempt, unlike POIPrivacy/MainTUL/
LSTM-TrajGAN), and per the task's own explicit license to use judgment
here, this baseline is a clean in-house implementation of the Planar
Laplace mechanism **directly from the original 2013 paper's own formula**
(Theorem 3 / Section 4.2) -- the paper specifies a complete, three-step
algorithm (sample angle uniformly, sample radius via a closed-form inverse
CDF, convert to a location offset), making a direct implementation both
faster and more verifiable than vetting a third-party implementation of
unclear provenance.

### `puffin/baselines/gi_dp.py`

`planar_laplace_noise(epsilon, n, rng)`: samples `n` independent
(dx, dy) offsets in meters via `theta ~ Uniform(0, 2*pi)`,
`r = -(1/epsilon) * (W_{-1}((p-1)/e) + 1)` for `p ~ Uniform(0,1)`, where
`W_{-1}` is the lower real branch of the Lambert W function
(`scipy.special.lambertw(..., k=-1)`) -- the exact inverse-CDF sampler
derived from the paper's own CDF, `C_eps(r) = 1 - (1+eps*r)*exp(-eps*r)`
(cross-checked against the same CDF stated in GeoPrivacy's README,
independently derived). `perturb_latlon` converts the meter offset to a
(lat, lon) degree offset via the standard local equirectangular
approximation (111,320 m/degree latitude; longitude scaled by
`cos(latitude)`), the same conversion used by deployed GI
implementations such as the Location Guard browser extension.
`run_gi_dp(df, epsilon, seed)` applies this independently to every row of
a unified-schema dataframe, re-tessellates `cell_id` from the perturbed
(lat, lon) via `assign_grid_cells` (same reasoning as `trajgan.py`), and
copies `weekday`/`hour` through unchanged (the published mechanism has no
temporal component -- it privatizes location only).

**Documented, deliberate scope limitation** (not a bug): this is the base,
single-location mechanism applied independently, point-by-point, to every
point of a real trajectory, since every dataset in this project is a
trajectory. The noise draws are independent across points, so the whole
*trajectory* is not epsilon-geo-indistinguishable as a unit -- only each
point is, in isolation. Composing a genuine trajectory-level guarantee
requires exactly the kind of budget-splitting / temporal-correlation-aware
extension privkit's adaptive variant exists for, which was explicitly
ruled out above as out of scope ("the classical mechanism"). Flagged
directly in the module docstring rather than silently assumed away.

**Tested**: `tests/test_gi_dp.py` (4 checks, all pass) -- the Planar
Laplace radius sampler's empirical mean matches the distribution's own
closed-form `E[r] = 2/epsilon` to within 0.2% at eps in {0.001, 0.01, 0.1}
(200,000 draws each: 2004.6m/200.4m/20.0m vs. theoretical
2000/200/20m) -- a real, manually-computable correctness check, not just
"did it run"; noise direction is unbiased (mean dx/dy ~0); the privacy
knob behaves monotonically (smaller epsilon -> strictly larger mean
perturbation distance: 19.96m / 199.63m / 1996.26m at eps=0.1/0.01/0.001);
and `run_gi_dp` returns a schema-valid dataframe, actually moves lat/lon,
and leaves user_id/traj_id/weekday/hour byte-identical to the input.

**Validated on real GeoLife** (`data/processed/geolife/unified.parquet`,
24,876,961 rows, 181 users, via `scripts/smoke_test_phase5_gidp.py`, run
across epsilon in {0.001, 0.01, 0.1} -- i.e. progressively less noise /
less privacy -- ~105-112s each on the full dataset, no GPU needed): the
empirical mean perturbation distance matched the mechanism's own
theoretical `E[r] = 2/epsilon` almost exactly at every epsilon --
1999.7m vs. 2000.0m (eps=0.001), 200.0m vs. 200.0m (eps=0.01), 20.0m vs.
20.0m (eps=0.1) -- confirming the implementation is correct on real data,
not just the synthetic toy data `tests/test_gi_dp.py` uses. The real
GeoLife tessellation has 85,074 distinct `cell_id` values; the perturbed
output's own re-tessellated vocabulary grows with noise, as expected:
349,900 distinct cells at eps=0.001 (~2km average perturbation spreads
points across a much larger area than the real trajectories ever
covered) vs. 86,109 -- close to the original 85,074 -- at eps=0.1's
~20m average perturbation.

## Phase 4's `floor_loss`: fixed-threshold inertness on GeoLife

**Finding (PI-directed check, post-Checkpoint-04c):** GeoLife's
`floor_loss` outputs were bit-identical across lambda4 in {0.3, 1.0,
3.0} in the Phase 4 grid -- a signature of the same inertness this
project already found once before in PUFFIN v1's original `ed` loss
(Checkpoint 0). Diagnosed directly rather than assumed: added
per-step gradient-norm logging (`scripts/diagnose_floor_loss.py`) and
confirmed `floor_loss`'s in-model adversary confidence never exceeds
~0.008 on GeoLife (chance is 1/180=0.0056) across 5 full epochs,
because the encoder's own adversarial training successfully suppresses
it near chance by design -- the same min-max dynamic this project has
relied on throughout. The original fixed `floor=0.3` default was
~54x higher than that achievable range, so `relu(confidence - 0.3)`
was exactly zero for every sample, every batch, every lambda4 value --
not a broken gradient computation (verified: `torch.autograd.grad`
on a deliberately lowered test floor DOES return a nonzero gradient),
just an unreachable threshold.

**Fix**: `floor_loss`'s `floor` parameter is now `floor_multiplier`
(default 1.0), with the actual floor computed as
`floor_multiplier / total_users` (total_users inferred from
`entropy_table`'s length) instead of a fixed absolute value -- so the
threshold scales with each dataset's own chance level rather than an
arbitrary constant. Confirmed via the same gradient-norm diagnostic:
at `floor_multiplier=1.0` (exactly chance level), roughly half of each
batch's samples exceed the floor, giving a genuinely nonzero gradient
throughout training (previously exactly zero). Honest caveat, stated
rather than oversold: the natural gradient scale at this threshold is
still small (~1e-7 per step) relative to the other loss terms even
once active, which may limit floor_loss's practical effect regardless
of lambda4 -- this is a property of how little headroom the in-model
adversary's confidence has above chance in this training setup, not
something the threshold choice alone can fully overcome.

**Tested**: `tests/test_phase4_losses.py` gained
`test_floor_loss_uses_chance_relative_threshold_not_fixed` (confirms
the floor scales with `total_users`, so a 180-class and a 1000-class
problem both get a reachable threshold) alongside the existing
`test_floor_loss_zero_below_floor_positive_above` (updated for the
renamed parameter). All pass.

**Rerun**: GeoLife's floor_loss points (lambda4 in {0.3, 1.0, 3.0})
were retrained with the fix as part of the PI-approved batch covering
the 6 metric-correction/utility-logging runs, the no-privacy
reference, and the >=3-encoder-seed upgrade -- see
`reports/04c_formulations_naive_standard.md`'s rebuild for the
corrected numbers.


## Checkpoint 04e: the approved batch's 20-epoch training is undertrained for every mechanism, not just no_privacy (2026-10-27)

**Found while diagnosing the PI's flagged anomaly**: on GeoLife, ed_soft's
utility (0.654) exceeded the no-privacy reference's utility (0.623), which
should be structurally impossible if no-privacy is a genuine ceiling. A
training-curve diagnostic (`scripts/diagnose_no_privacy_ceiling.py`, one
seed, 40 epochs instead of the batch's 20) found the real cause: ALL THREE
mechanisms were still rising substantially at epoch 20, with their relative
ORDER changing between epoch 20 and 40 --

| mechanism | utility @ epoch 20 | utility @ epoch 40 |
|---|---|---|
| no_privacy | 0.611 | 0.735 |
| plain | 0.473 | 0.719 |
| ed_soft (λ4=1.0) | 0.680 | 0.740 |

At epoch 20, `ed_soft > no_privacy > plain` -- the exact anomalous ordering
the PI flagged. By epoch 40 the three are close together (0.719-0.740),
consistent with no_privacy being a (barely) genuine ceiling once properly
converged. `plain` also shows a long cold-start (~10 epochs stuck near
chance before utility moves at all), visible in the same log.

**This means the anomaly was not a no_privacy-specific tuning bug -- it is
differential convergence SPEED across mechanisms at a shared, too-low fixed
epoch count (20), with no early stopping anywhere in
`run_phase4d_approved_batch.py` (confirmed: a fixed `for epoch in
range(epochs)` loop, no validation-based check).** A mechanism that happens
to warm up faster looks artificially better at epoch 20 regardless of its
actual steady-state utility, and vice versa.

**Fixed for this round's new training only**: Checkpoint 04e's reference-
curve sweep, no_privacy refit, and GeoLife 10-encoder-seed rerun
(`scripts/launch_phase4e_batch.sh`) all use `--mopae-epochs 40` (the point
where the diagnostic's curves visibly flatten), not 04d's 20.

**Not fixed**: the original 21-run Checkpoint 04d batch remains at 20
epochs. Retraining all 21 at 40 epochs is a significant compute decision
(roughly doubling that batch's cost) that is flagged here for the PI's
sign-off rather than taken unilaterally -- see `reports/04e_reference_curve.md`.
Every utility comparison already drawn from 04d's 20-epoch data (including
the no-privacy-exceeded-by-ed_soft anomaly itself, and any utility-matched
comparison in the leveling-down check) should be read with this confound in
mind until/unless that rerun happens.


## Foursquare CUDA OOM: unbatched encoder forward pass in attacker-in-the-loop, and an orchestrator that died on it (2026-10-30)

**Found**: the attacker-in-the-loop batch (launched 09:46 AM, 24 jobs) got
through all 8 GeoLife jobs cleanly, then crashed immediately on the first
Foursquare job with `torch.OutOfMemoryError: CUDA out of memory. Tried to
allocate 16.01 GiB`. The orchestrator script (`launch_attacker_in_the_loop.sh`)
used `set -e`, so that one job's failure killed the whole script, leaving
16 of 24 jobs unqueued with GPUs sitting idle.

**Root cause**: `refit_fresh_attacker_head()` (`puffin/fairness/attacker_in_the_loop.py`)
ran `feat = encoder(cell, weekday, hour)` on the FULL training set in one
unbatched forward pass, before mini-batching the classifier-head training
that follows it. GeoLife has ~36,200 training windows and got through
fine; Foursquare has ~216,600 -- about 6x more -- through an LSTM
encoder, which is what exhausted memory. Every other attacker in this
codebase already chunks exactly this kind of full-dataset encoder
forward pass via `puffin.attacks.reid._batched_forward`; this one call
site was missed when the formulation was built.

**Fixed**: `refit_fresh_attacker_head` now calls `_batched_forward` for
the encoder pass (new `encoder_forward_batch_size` parameter, default
4096). Grepped the whole codebase for other direct `encoder(...)` calls
on a full split rather than a mini-batch slice: found the SAME pattern
in `eval_in_model_confidence` and `eval_utility`
(`scripts/run_phase4d_approved_batch.py`) -- both ran the encoder on the
full TEST set unbatched. These were lower-risk before (called once, at
the very end of training) but became a real risk once early stopping
(see below) calls an equivalent validation check every epoch. Both
fixed the same way. `scripts/run_phase4_formulations.py`'s `eval_utility`
has the identical bug but that script is no longer part of the active
training pipeline (see below) and was not fixed.

**Tested**: `tests/test_attacker_in_the_loop.py` gained
`test_refit_fresh_attacker_head_batches_the_encoder_forward_pass` -- a
Foursquare-sized (220,000-row) input through an encoder wrapped to
assert it is never called with more rows than the configured batch
size, so this regresses loudly (an assertion failure) rather than
silently if the unbatched call pattern is ever reintroduced. All
existing tests across the whole suite re-run clean (no regressions).

**Orchestrators made fault-tolerant**: every orchestrator shell script
now runs without `set -e`. Each job's exit status is checked
individually (`wait "$pid"` per job, not just the batch), a failure is
logged with its exit code (the job's own log has the full traceback)
and the queue continues regardless, and a `failed_jobs_summary.txt` is
written at the end. Applied to the new `launch_retraining_to_convergence.sh`
(written with this pattern from the start) and to
`launch_attacker_in_the_loop.sh` (rewritten; not relaunched yet). The
already-finished `launch_phase4e_batch.sh` was not retrofitted (its work
is now fully superseded by the retraining-to-convergence batch and it
will not be run again).

## Training protocol: early stopping on a chronological validation split, every mechanism (2026-10-30)

**Problem this closes**: the earlier diagnostic (`scripts/diagnose_no_privacy_ceiling.py`)
found every mechanism undertrained at a fixed epoch count, with
differential convergence speed confounding utility comparisons across
mechanisms. The ad-hoc fix at the time (doubling 20 epochs to 40 for
some new training only) was itself not justified by a per-mechanism
convergence check and left most of the Phase 4 grid at the original,
unconverged 20 epochs.

**Fixed properly**: `scripts/run_phase4d_approved_batch.py` is now the
single, unified driver for every fairness formulation in this project
(it absorbed `lipschitz`, `variance`, `group_dro`, `per_user_minimax`,
and `group_adversarial` from the now-superseded
`run_phase4_formulations.py`, which used the RETIRED adaptive-attacker
evaluation standard and is no longer part of the active pipeline).
Every mechanism trains up to `--mopae-epochs` (now a ceiling, default
80) and early-stops on a per-user CHRONOLOGICAL validation split carved
from the training windows -- the last `--val-frac` (default 0.1) of
each user's train windows BY TIME, via `puffin.attacks.reid`'s existing
`_chronological_val_split` (already used for attacker-side early
stopping; this is the first time it is applied to mechanism training
too). `--patience` (default 8) is identical across every mechanism, so
no formulation is advantaged or disadvantaged by how many epochs it
happens to run. The held-out TEST split is never touched until final
evaluation. `group_adversarial`'s own training step (no `id_recognizer`;
a per-decile-group adversary instead) is handled via the same generic
early-stopping loop, just with its own module set for the
best-validation-utility snapshot.

**Not fixed/out of scope this round**: the five previously single-
encoder-seed formulations remain single-seed in this round's retrain
(they get the new early-stopping protocol, not more seeds) -- flagged
in `PROGRESS.md`, not resolved unilaterally.


## GeoLife's 85,074-location vocabulary OOM'd eval_utility too, after the Foursquare fix (2026-10-30)

**Found**: the first fix (CPU-resident best-state snapshots, see above)
did not resolve GeoLife specifically -- all 7 of GeoLife's item-1 jobs
failed again, with the identical `torch.OutOfMemoryError: ... Tried to
allocate 10.51 GiB` at the exact same call site
(`eval_utility`'s `_batched_forward`), on the very first encoder seed's
final evaluation, with training itself (including every per-epoch
validation check) completing without incident. Meanwhile Foursquare's
7 item-1 jobs and the first 2 of Gowalla-SF-Bay's all succeeded.

**Root cause**: GeoLife has 85,074 distinct location cells (a
fine-grained continuous-GPS grid), vs.\ Foursquare's 1,668 and
Gowalla-SF-Bay's ~2,100 -- a 40-50x larger vocabulary.
`next_predictor`'s output width is `total_locations`, so each row of
logits is 85,074 floats wide on GeoLife. `_batched_forward`'s pattern
(accumulate every chunk's output in a Python list, then `torch.cat`
them all at the end) means the memory needed for the FINAL
concatenated result is the same regardless of chunk size -- smaller
chunks reduce peak memory during each individual forward call, but not
the total memory required to hold the complete `[n_test, 85074]`
logits tensor before it can be reduced to an accuracy number. This is
why training's own per-epoch validation checks (over only the
validation split, a smaller slice) survived while the final
test-set-sized evaluation did not, and why bigger-by-window-count
Foursquare/Gowalla-SF-Bay were fine -- their vocabulary, not their
window count, is what matters for this specific memory cost.

**Fixed**: new `_batched_accuracy` helper computes accuracy
incrementally -- each chunk's predictions are reduced to a match count
immediately and discarded, so the full `[n, total_locations]` tensor is
never materialized, only one `[batch_size, total_locations]` chunk at a
time (default `batch_size=1024`, smaller than `_batched_forward`'s
4096, for extra margin). `eval_utility` and `_val_utility` both use it
now. `eval_in_model_confidence` was not changed -- its output width is
`total_users` (181 for GeoLife), not `total_locations`, so it was never
at risk.

**Tested**: a synthetic check at GeoLife's exact vocabulary scale
(85,074 locations) confirms `_batched_accuracy` produces output
identical to a naive, unbatched computation (exact match, not just
close), and the full existing test suite + the 9-mechanism smoke test
both re-run clean. No orchestrator restart was needed: since each job
is a fresh `python` process launched by the (still-running) shell
orchestrator, every job launched after this fix landed on disk picked
it up automatically, including the GeoLife jobs still queued in items
2, 4, and 5.
