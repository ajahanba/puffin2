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
