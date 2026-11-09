## Current state (2026-11-08): all core baselines/datasets finished ahead of the Nov 9 hard stop; TrajGAN (GeoLife only) attempted same day, gated on a 10 PM start

**CPU thread oversubscription, found and fixed (2026-11-04).** Every
training process was spawning 120-131 OS threads on a 40-core box
(PyTorch/OpenBLAS/MKL default to uncapped pools); two concurrent jobs
caused catastrophic slowdowns (GPU at 0% utilization, CPU at
1000-1700%). Fixed by capping `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
`OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS` and calling
`torch.set_num_threads()`, all BEFORE numpy/torch import (env vars set
after import have no effect) -- applied to
`scripts/run_phase4d_approved_batch.py` and
`scripts/run_gi_dp_current_protocol.py`. Confirmed working (threads
dropped to 35-47 per job); the apparent continued slowness on
YJMob100K/Gowalla-SF-Bay afterward turned out to be real compute cost
(see per-phase timing below), not a residual oversubscription bug.

**GI-DP baseline (Geo-Indistinguishability, Planar Laplace), current
protocol: all 15 jobs done** (3 datasets x 5 epsilon in
{0.0005,0.001,0.002,0.005,0.01}, 3 encoder seeds, 5 attacker seeds each,
same strongest-per-user attacker as every other mechanism).
`scripts/run_gi_dp_current_protocol.py`. Three real bugs found and
fixed before/during this run:
- `assign_grid_cells` built an independent fresh cell-id mapping per
  call -- comparing original vs. perturbed `cell_id` was meaningless
  without reusing the mapping. Fixed via an `existing_mapping` param
  (`puffin/data/tessellation.py`), regression test added.
- GI-DP's output never carried the `split` column through, crashing
  window-building. Fixed in `puffin/baselines/gi_dp.py`.
- **Output-vocabulary bug (2026-11-06, PI-caught)**: the next-location
  predictor's classification head was sized to the perturbed release's
  EXTENDED cell vocabulary (which can be 6x+ larger than the original
  at small epsilon, e.g. GeoLife eps=0.0005: 85,074 -> 544,552 cells),
  but true next-location labels only ever range over the ORIGINAL
  vocabulary. Fixed by splitting `embedding_vocab` (input, extended --
  must cover the perturbed release's cell ids) from `output_vocab`
  (head, stays at the original size). **Confirmed**: `output_vocab_size`
  in every one of the 15 summary.json files equals the dataset's fixed
  original vocabulary at every epsilon (GeoLife 85074, Foursquare 1668,
  Gowalla-SF-Bay 2070), identical across the whole epsilon sweep;
  `embedding_vocab_size` correctly shrinks as epsilon grows (less
  noise, fewer new cells).
- Zero dropped points confirmed at every (dataset, epsilon): grid
  extension is used throughout, never clamping/dropping, per the PI's
  instruction. Bounds/new-cell report:
  `reports/figures/gi_dp_analysis/bounds_and_new_cells_report.json`.

**GI-DP moved from CPU to GPU1 mid-run (2026-11-06), per the PI's own
prior contingency** ("move remaining jobs to a GPU once one frees up
and CPU pace risks missing Nov 8"): GPU1 had been idle for 23+ hours
after Task 2 finished, and newly-added per-architecture timing (below)
showed GRU -- not GBT, which stays on CPU/numpy regardless of
`--device` -- dominated Foursquare's cost (1640s/seed vs. GBT's 102s).
Moving to GPU1 was a 6-12x speedup: remaining jobs dropped from
6-9.6h/job (CPU) to 39min-1h16min/job (GPU). All 15 jobs finished
2026-11-07, 5:08 AM, comfortably inside the Nov 8 deadline.

**Per-architecture attacker timing, new instrumentation**
(`puffin/attacks/reid.py`, both `train_raw_attacker_strongest_per_user`
and `train_attacker_on_frozen_encoder_strongest_per_user`): print-only,
no return-signature change. Revealed the real driver of YJMob100K's
slow pace (never a thread-oversubscription bug): for one YJMob100K job
(`ed_soft_lambda0.3`, 707 min total), encoder/MoPAE training was only
~19 min (2.7%); **GBT (gradient-boosted trees) alone was ~84%** (raw
attacker GBT ~32 min + 3x post-attacker GBT ~187 min each). For
Foursquare's GI-DP jobs the pattern inverted -- GRU dominated, GBT was
cheap -- confirming the bottleneck architecture is dataset-dependent,
not a fixed property of the pipeline.

**YJMob100K: all 10 jobs done** (2026-11-08, 9:28 AM), reordered mid-run
per the PI's instruction (2026-11-04): after `plain_lp0.1`, ran
`ed_soft lambda4=1.0` next, then only the plain lambda3 points needed
to bracket its utility against the known grid (sequential search,
stopping the instant crossed, never extrapolated -- needed exactly 1
extra point, `lp0.2`, utility 0.1805 <= target 0.1826 <= lp0.1's 0.1975),
then `ed_soft lambda4={0.3,3.0}`, then the 3 leftover plain points
(`{0.3,0.5,0.8}`, "if time allows") -- all of which ended up fitting.
Implemented as `scripts/launch_yjmob100k_phase2.py`, handed off from the
original orchestrator via a watcher that killed it the instant
`plain_lp0.1`'s summary.json appeared (`scripts/yjmob_phase2_watcher.sh`).
One flagged seed (`plain_lp0.8`, seed 2: naive 0.0695 -> audited 0.0976,
`artifact_fixed`); `_corrected` copy written per the standard plateau-
audit procedure.

**Group-DRO (GeoLife, 3 seeds, lambda4 in {0.3,1.0}): done**
(2026-11-05), inserted into the Task2/GPU1 queue between Gowalla-SF-Bay
and Foursquare Lipschitz via the same watcher-handoff pattern
(`scripts/task2_groupdro_watcher.sh`). utility_mean 0.742 (lambda4=0.3),
0.743 (lambda4=1.0).

**Task2 Lipschitz queue: Foursquare (both similarity bases, 3 seeds)
and the remaining Gowalla-SF-Bay location-distribution points: all
done** (2026-11-05). GeoLife's 10-seed Lipschitz runs (both bases)
already existed from the original Task2 pass.

**Protocol tags backfilled for every new directory, including GI-DP**
(`scripts/backfill_protocol_tags.py`, extended to also scan
`reports/figures/gi_dp/` -- it previously covered only
`retraining_to_convergence/` and `task2_formulations/`). Verified: all
35 of this session's new run directories (10 YJMob100K incl.
`_corrected`, 15 GI-DP, 2 Group-DRO, 9 Foursquare/Gowalla Lipschitz) tag
`protocol_ok: true` (patience=8, val_frac=0.1, min_epochs=20, >=1
encoder seed, descendant of the min_epochs-burn-in-fix commit).
`yjmob100k_plain_lp0.8_auditseed2` (the raw plateau-audit rerun
artifact, min_epochs=40 by design) correctly flags `protocol_ok: false`
-- it is superseded by `_corrected` and was never meant to be read
directly. Did NOT additionally run the separate contamination-audit
`usable`/`usable_note` merge step from earlier phases for these new
directories -- it predates this round's plateau-audit machinery and its
applicability to these runs wasn't established; flagging this gap
rather than silently applying or skipping it.

**TrajGAN (GeoLife only), current protocol: attempted 2026-11-08,
gated on a 10 PM start.** Utility/risk protocol decided with the PI
given a structural asymmetry in the existing `run_trajgan()`
(`puffin/baselines/trajgan.py`): the generator trains once on the REAL
train split and synthesizes only a counterpart of the test split (no
synthetic train exists). Predictor trains on the real train windows,
evaluated on the synthetic test release against true labels; the
strongest-per-user attacker gets an internal 50/50 chronological split
carved from the synthetic release itself (no synthetic train to use
instead). New driver: `scripts/run_trajgan_current_protocol.py`, reusing
the same cell-mapping-reuse fix as GI-DP (the module's own internal
`assign_grid_cells` call does not reuse the original mapping) and the
same window-alignment assertion. Smoke test surfaced that the existing
module's trajectory-grouping prep step (`_trajectory_groups`, a
Python-level `groupby` loop, pre-existing code) is far slower than
expected on GeoLife's full dataset -- a fixed cost paid once per
invocation regardless of epoch count. [STATUS AT TIME OF WRITING: see
the next report for whether caching this to disk got the full run
(generator + standard 5-seed x 3-architecture attacker) started on GPU
by the PI's 10 PM gate; if not, this is recorded as not evaluated for
this submission, per the PI's explicit instruction, and no Foursquare/
Gowalla TrajGAN was attempted.]

**TrajGAN: NOT EVALUATED for this submission (2026-11-09), per the PI's
explicit contingency.** Caching the trajectory-grouping step to disk
(`puffin/baselines/trajgan.py`'s `_cached_trajectory_groups`, keyed by
`cache_dir`/`cache_key`) was implemented and works, but a 2-epoch,
1-encoder-seed smoke test on GeoLife revealed the real blocker: **GAN
training itself took 7.18 hours for 2 epochs** (`final_d_loss=0.0427`,
`final_g_loss=820.67`, well short of convergence). At that rate a real
run (default 60 epochs) would take on the order of 9 days, not hours --
infeasible regardless of caching, and the PI's 10 PM/2026-11-08 gate for
starting the full run (generator + standard 5-seed x 3-architecture
attacker) on GPU passed without it starting. Separately, the same smoke
test surfaced a real bug in the driver's protocol design: the window-
alignment assertion between the real test split and the synthetic
release failed (`run_trajgan` truncates every trajectory to its first
150 points before generating a synthetic counterpart, which can make
the set of SL=10 windows built from the synthetic release diverge from
the real test split's own window set for any user with a longer
trajectory -- not yet root-caused further, since the GAN wall-clock cost
alone already rules out a full run in the remaining time). Per the PI's
instruction, no Foursquare or Gowalla-SF-Bay TrajGAN was attempted.
`scripts/run_trajgan_current_protocol.py` and the caching fix are
committed for future work but produced no usable result this round.

**Gowalla-SF-Bay Lipschitz, seeds 3-4 added (2026-11-08/09), on the GPU
freed up once GeoLife's current-protocol Lipschitz runs were confirmed
to already exist (10-seed, superseding the requested 3-seed ask).**
Both similarity bases, lambda4 in {0.3,1.0,3.0}, encoder-seeds "3,4"
(additive, not replacing the existing 0,1,2 seeds) --
`scripts/launch_gowalla_seeds34.sh`, output dirs suffixed `_seeds34`.
5 of 6 done as of this entry, last one (location_distribution,
lambda4=3.0) in progress, ETA ~10:30 AM. Note for whoever uses these:
they are a SEPARATE 2-seed set, not yet merged with the existing 3-seed
(`_3seed`) runs into a combined 5-seed aggregate -- that merge is not
done.


## Superseded (2026-11-04): paper reframed around individual fairness; Task 1 done (with a major correction), Task 2 in progress, target Nov 12


# PUFFIN v2 — Progress

## Superseded (2026-11-04): paper reframed around individual fairness; Task 1 done (with a major correction), Task 2 in progress, target Nov 12

**The paper's focus changed (2026-11-02, PI instruction).** This is now a
paper about individual fairness of privacy protection: do users with
similar mobility receive similar protection, and does PUFFIN (the
entropy-difference fairness term, `ed_soft`) make protection fairer? The
evaluation machinery (external attacker, seeds, decile risk, protocol
tagging) supports that question; it is not the contribution. Census/group
analysis is out of scope for this submission.

**`paper/` has been replaced and is frozen.** A parallel Claude session
rebuilt `paper/main.tex` from the original PUFFIN manuscript (commit
`532ffaf`); the previous draft (disparate-risk framing, Table 7, the
PoPETs-template work) is preserved in `paper_legacy/`. Per the PI's
explicit instruction, **do not edit anything under `paper/` until told
it's in place.**

**Task 1 (individual-fairness analysis) is done for all three datasets,
with one major mid-course correction.** `reports/06_individual_fairness.md`:
four similarity definitions computed from training-split data only
(entropy, 1-JS-distance between visitation histograms, Hausdorff on
(subsampled) lat/lon, radius of gyration; `puffin/metrics/
individual_fairness.py`), the individual-fairness metric at
delta=0.8/top-10%-similar-pairs and epsilon=0.2/curve, dose-response,
all bootstrapped over seeds and users.

The correction: **the raw absolute IF-rate mostly tracks overall privacy
level (mean risk), not genuine individual fairness** (Foursquare
lambda3=0.5: utility 0.06, mean risk 0.03, raw rate 0.98 under every
similarity definition -- that 0.98 is mostly "this point gives a lot of
privacy"). Added a scale-free ratio (top-10%-pairs mean |risk_i-risk_j|
divided by the same quantity over ALL pairs) and, after a further PI
correction, a THIRD framing: report all of (a) the raw absolute
mean-|risk_i-risk_j| over top-10% pairs, (b) that divided by mean risk,
and (c) the scale-free ratio, side by side -- (c) alone can be fooled by
a mechanism that simply narrows the population's own risk spread, which
is a separate effect from individual fairness. With the scale-free
correction, GeoLife's "plain is more fair than no-privacy" finding mostly
does NOT survive (3 of 4 definitions turn slightly negative); a smaller
real effect survives on Foursquare/Gowalla-SF-Bay for entropy/location-
distribution/trajectory-shape; radius-of-gyration shows essentially no
genuine signal anywhere.

**A second correction, also from PI review against the per-user CSVs**:
an earlier claim that PUFFIN trained on location-distribution similarity
"makes trajectory-shape/radius-of-gyration fairness worse than random"
was wrong -- no-privacy and plain Mo-PAE show the SAME ratios (~1.3-1.4)
on GeoLife; it is a baseline property of that dataset/definition pair,
not a PUFFIN effect. The location-trained variant is statistically
indistinguishable from plain Mo-PAE on every metric. A gradient-magnitude
check confirms why: `ed_soft_location`'s loss (MSE over an 85,075-entry
near-uniform probability vector for GeoLife) produces a gradient **~3x10^8
times smaller** than `ed_soft`'s (MSE over a single entropy scalar) at the
same lambda4 -- the loss term is providing essentially no training signal
as currently scaled. This needs a rescaling fix before "PUFFIN with a
different similarity basis" can be tested honestly; not yet done.

**The GeoLife lambda3 sweep (plain Mo-PAE reference curve) is complete**:
extended with {1.2, 2, 3} beyond the original {0.02,...,0.8}, 10 seeds
each, stopped per the PI's rule (crossed utility<=0.70 at lambda3=2.0,
ran lambda3=3.0 as the one extra point, stopped). This makes GeoLife's
matched-utility PUFFIN-vs-curve comparisons resolvable for the first time
(utility now ranges ~0.46-0.76, bracketing PUFFIN's own 0.71-0.74) --
Table 7's frontier-test data and report 06 Section 5 are being redone
with the extended curve; not yet landed as of this entry.

**Plateau-vs-genuine-collapse audit, new infrastructure**
(`puffin/eval/plateau_audit.py`): at higher privacy weights a seed sitting
at a low-utility plateau may be a genuine collapse, not the known
cold-start artifact -- the PI's instruction is to rerun each flagged seed
with `min_epochs` doubled rather than assume either way. lambda3=2.0's one
flagged seed was confirmed `artifact_fixed` (0.062 at 20 epochs -> 0.665
at 80 epochs once given more floor); a corrected copy of that run
(`geolife_plain_lp2_10seed_corrected`) replaces the contaminated seed's
full per-user data, not just its utility scalar.
`apply_audit_corrections` does this merge generically. lambda3=3.0's
flagged seed is being audited the same way.

**Task 2 (new training) is in progress on GPU1** (GPU0 is free, the
sweep finished): Lipschitz (both entropy and location-distribution
similarity bases, three datasets, lambda4 in {0.3,1.0,3.0}) and PUFFIN-
with-location-distribution (GeoLife only, done, see above). Found and
fixed a real protocol bug while building this: GeoLife jobs were
launched with `--respect-traj-boundary`, inconsistent with every other
GeoLife job this round; caught after 1 job completed and 1 was mid-run,
both deleted and rerun. One unexplained anomaly: a ~12x overnight
slowdown hit both GPUs (a ~70-min job took 868 min); system load was
back to normal and the next job's pace recovered by the time it was
checked, cause not identified.

**Every run directory is protocol-tagged** (`scripts/backfill_protocol_tags.py`,
now covers both `reports/figures/retraining_to_convergence/` and
`reports/figures/task2_formulations/`) and only `usable: true` runs feed
any analysis or table.

## Superseded (2026-11-03): retraining-to-convergence batch running; superseded by the fairness-reframing pivot below

**The attacker-in-the-loop batch crashed on its first Foursquare job**
(2026-10-30, ~11:00 AM) with a CUDA OOM, and its orchestrator's `set -e`
killed the whole script with 16/24 jobs still unqueued. Root cause and
fixes, full detail in `CHANGES_FROM_ORIGINAL.md`:
- `refit_fresh_attacker_head()` ran the encoder forward pass on the
  FULL training set in one unbatched call (fine on GeoLife's ~36k
  windows, a 16GB single allocation on Foursquare's ~216k through an
  LSTM). Fixed via `puffin.attacks.reid._batched_forward`, the same
  chunking helper every other attacker in this codebase already uses.
  Grepped for the same pattern elsewhere: found and fixed the identical
  bug in `eval_in_model_confidence`/`eval_utility`
  (`scripts/run_phase4d_approved_batch.py`) -- lower-risk before
  (called once, at the end), now a real risk since early stopping (see
  below) calls an equivalent check every epoch.
- New regression test: `tests/test_attacker_in_the_loop.py::test_refit_fresh_attacker_head_batches_the_encoder_forward_pass`
  (Foursquare-sized input, asserts the encoder is never called with
  more rows than the configured batch size). Full existing test suite
  re-run clean, no regressions.
- **Every orchestrator is now fault-tolerant**: no `set -e`; each job's
  exit status is checked individually, failures are logged with their
  exit code and the queue continues, a `failed_jobs_summary.txt` is
  written at the end. Applied to the new
  `scripts/launch_retraining_to_convergence.sh` (built this way from
  the start) and to `scripts/launch_attacker_in_the_loop.sh` (rewritten,
  not relaunched yet).

**Training protocol changed project-wide: early stopping on a
chronological validation split, same patience for every mechanism.**
`scripts/run_phase4d_approved_batch.py` is now the single, unified
driver for every fairness formulation studied in this project -- it
absorbed `lipschitz`, `variance`, `group_dro`, `per_user_minimax`, and
`group_adversarial` from `run_phase4_formulations.py` (which used the
RETIRED adaptive-attacker standard and is no longer part of the active
pipeline). Every mechanism trains up to `--mopae-epochs` (now a
ceiling, default 80) and early-stops (`--patience 8`, `--val-frac 0.1`)
on each user's chronologically latest train windows, via
`puffin.attacks.reid`'s existing `_chronological_val_split` (previously
attacker-side only, now applied to mechanism training too). This
replaces both the original fixed-20-epoch protocol and the ad-hoc
"doubled to 40" fix from the previous round.

**Running now** (`scripts/launch_retraining_to_convergence.sh`, 79
jobs, both GPUs, started 2026-10-30 ~11:44 AM, target **November 12**),
in priority order:
1. plain (privacy weight swept -- the Pareto reference curve) +
   no-privacy, all 3 datasets, 3 encoder seeds (21 jobs).
2. GeoLife plain vs.\ ed\_soft, 10 encoder seeds (4 jobs).
3. ed\_soft on Foursquare and Gowalla-SF-Bay, 3 encoder seeds (6 jobs).
4. floor, GeoLife only, 3 encoder seeds (3 jobs).
5. The five single-encoder-seed formulations, all 3 datasets, 1 seed
   each, now under the same early-stopping protocol (45 jobs).

**Gowalla subset changed to gowalla\_sf\_bay (2,797 users) for this and
all future new training**, not `gowalla_nyc` (1,517, used throughout
04d/04e) -- per the PI's explicit, repeated naming for this round and
for attacker-in-the-loop. 04d/04e's existing gowalla\_nyc results are
untouched.

**Attacker-in-the-loop: NOT resumed.** Per instruction, it will be
rerun once the retraining batch is done (or sooner, on any GPU that
frees up), under the same early-stopping protocol, in the order
GeoLife -> Gowalla-SF-Bay -> Foursquare. The 8 GeoLife runs that
completed before the crash (fixed 40 epochs, 3 seeds) are archived at
`reports/figures/attacker_in_the_loop_early_signal_fixed40ep/` and are
**not used in the paper** -- early signal only:

| config | utility | decile-0 risk |
|---|---|---|
| plain Mo-PAE reference (40ep, 10 seeds) | 0.731 | 0.359 |
| k=1, λ4=0.3, floor | 0.702±0.018 | 0.308 |
| k=1, λ4=0.3, group\_dro | 0.701±0.024 | 0.320 |
| k=1, λ4=1.0, floor | 0.696±0.016 | 0.349 |
| k=1, λ4=1.0, group\_dro | 0.696±0.002 | 0.307 |
| k=5, λ4=0.3, floor | 0.688±0.006 | 0.346 |
| k=5, λ4=0.3, group\_dro | 0.739±0.010 | 0.325 |
| k=5, λ4=1.0, floor | 0.703±0.020 | 0.341 |
| k=5, λ4=1.0, group\_dro | 0.742±0.001 | 0.309 |

Every one of the 8 configs shows lower decile-0 risk than the plain
reference (0.307-0.349 vs.\ 0.359), with comparable-to-higher utility --
a consistent, if modest, pattern worth confirming once retrained
properly; not a result, no CIs, 3 seeds only, fixed epoch count.

**A second, GeoLife-specific OOM surfaced once the retraining batch
actually launched** (all 7 of GeoLife's item-1 jobs failed, Foursquare's
7 and Gowalla-SF-Bay's all succeeded) -- the fault-tolerant orchestrator
worked exactly as designed, logging each failure and moving on rather
than dying. Root cause: GeoLife has 85,074 distinct location cells (a
fine GPS grid) vs.\ Foursquare's 1,668 and Gowalla-SF-Bay's ~2,100 --
`eval_utility`'s batched forward pass accumulated every chunk's
`[batch, 85074]` logits in a list before concatenating, and the final
concatenated tensor -- not the per-chunk peak -- is what exhausted
memory, which no amount of shrinking the chunk size would have fixed.
New `_batched_accuracy` helper reduces each chunk to a match count
immediately instead of concatenating logits; verified against a naive
computation at GeoLife's exact vocabulary size (exact match) and the
full test suite re-run clean. No orchestrator restart was needed --
each job is a fresh `python` process, so every job launched after the
fix landed on disk (including GeoLife's still-queued items 2/4/5) picks
it up automatically. Full detail in `CHANGES_FROM_ORIGINAL.md`.

---

**Checkpoint 04e is complete.** Full report: `reports/04e_reference_curve.md`
(5-line summary at top). Headline: **the GeoLife "ed\_soft dominates
plain" claim from 04d is retracted.** A clean, matched-epoch (40), 10-
encoder-seed paired bootstrap test shows a real, significant decile-0
risk reduction at λ4∈{0.3,1.0} with **no** significant utility gain
(a partial benefit, not dominance), and a significant utility *cost* at
λ4=3.0 (an ordinary trade-off). No formulation in the batch is shown to
dominate plain Mo-PAE once both epoch count and encoder-seed count are
controlled.

Also done: the plain-Mo-PAE reference curve (λ3 swept, 40 epochs) --
nearly flat on GeoLife, clean/monotonic on Foursquare, noisier on
Gowalla-NYC; the reused 20-epoch λ3=0.3 point is a visible outlier on
its own curve and is excluded from interpolation. No-privacy ceilings
corrected (GeoLife 0.623→0.742; Foursquare 0.280→0.214; Gowalla-NYC
0.322→0.251 -- the two drops both fit their dataset's own curve, not a
regression). Recalibrated leveling-down check rerun: 0/40 comparisons
exceed the seed-to-seed null (likely underpowered at 3 seeds). Paper
Sections 6-7 rewritten against these numbers, two new generated tables,
test-compiled clean (10 pages, zero errors). All committed/pushed
(`18d17e0`).

**Running now, low priority** (`scripts/launch_attacker_in_the_loop.sh`,
24 jobs, both GPUs, `nice -n 19`, started 2026-10-30 ~09:46 PDT): the
PI's queued attacker-in-the-loop formulation -- GeoLife, Foursquare,
**Gowalla-SF-Bay** (not `gowalla_nyc`, per the PI's explicit naming for
this formulation) × k∈{1,5} × λ4∈{0.3,1.0} × loss\_variant∈{floor,
group\_dro} × 3 encoder seeds, **40 epochs** (not 04d's original 20,
per 04e's undertraining finding -- no reason to repeat that mistake on
a new formulation). To be evaluated against 04e's reference curve once
complete (not yet analyzed).

**Still open / not resolved this round**: retraining the rest of 04d
(`ed_soft` on Foursquare/Gowalla-NYC, `floor` on GeoLife) at 40 epochs
is flagged for the PI's sign-off, not done unilaterally -- until then,
every curve-comparison involving those points is directional only. A
per-epoch convergence diagnostic was only run on GeoLife; Foursquare/
Gowalla-NYC's 40-epoch no-privacy ceilings are not independently
confirmed to be converged. A richer, per-decile leveling-down check
using GeoLife's 10-seed null (90 pairs vs. 6) was not run this round.

---

## Superseded (2026-10-27): the material below predates the PI's 04d review and the corrections above

**The approved 21-run batch (Checkpoint 04d) is complete, analyzed, and
reported on. This closes the "stop and report" checkpoint the PI set:
rebuild 04c's fronts with plain Mo-PAE and the no-privacy point
overlaid, run the leveling-down check, report.** Full writeup:
`reports/04d_approved_batch.md`. Headline finding: the no-privacy
reference shows the privacy mechanism itself *widens* the decile0/
decile9 residual-risk-disparity ratio on Foursquare (3.43x -> 16.02x)
and Gowalla-NYC (4.12x -> 7.26x) relative to an undefended
representation, flat on GeoLife (8.02x -> 8.08x, no effect either
way); raising ed_soft's fairness weight on Foursquare makes decile 0's
*absolute* post-mechanism risk get worse (0.487 -> 0.686) even as
aggregate utility rises (0.111 -> 0.233) and the RRD ratio falls
(16.02x -> 5.28x) -- a textbook leveling-down pattern, confirmed by the
formal check. The fixed floor_loss (see below) protects GeoLife's
already-best-protected decile 9 more as lambda4 rises, not decile 0;
RRD gets worse (8.19x -> 9.68x), not better. ed_soft on GeoLife is the
one case in the whole batch that genuinely dominates plain Mo-PAE
(higher utility AND lower decile-0 risk, both CIs non-overlapping) --
every other (dataset, formulation, lambda4) point is a trade-off, and
several are trade-offs against the party the mechanism exists to
protect. Analysis script (pure re-analysis, no retraining):
`scripts/analyze_04d_approved_batch.py`, output in
`reports/figures/phase4d_analysis/`. `reports/04c_formulations_naive_standard.md`
has a dated 2026-10-27 superseded note pointing here.

**Paper fully updated to match**: `paper/sections/residual_risk_disparity.tex`
and `paper/sections/phase4_outline.tex` (previously left as blocked
outlines) now have real prose and two new script-generated tables
(`paper/tables/04d_decile_reduction.tex`, `04d_inmodel_vs_external.tex`,
via `paper/scripts/generate_figures.py` -- regenerate, never hand-edit).
Test-compiled locally (pdflatex x2 + bibtex, 9 pages, zero errors, zero
undefined references) before committing.

**floor_loss's in-model-vs-external mechanistic comparison** (added to
the 04d driver earlier, per the PI's request) is now written up in
both the report and the paper: in every dataset and every decile, the
in-model adversary's confidence sits far below the external attacker's
accuracy on the identical features, and its own decile-to-decile range
is far smaller. GeoLife is the cleanest case (in-model confidence
1.5x-4.5x chance across all 10 deciles vs. external's 8x-63x chance) --
the direct explanation for why floor_loss's gradient could never
engage before the fix, and why even the fixed version still can't
reliably single out decile 0. Foursquare and Gowalla-NYC show a
smaller, decile-0-specific in-model signal (not perfectly flat there),
still 7-9x weaker than the external attacker at that same decile.

**Attacker-in-the-loop fairness: built, smoke-tested, queued -- NOT
launched.** Per the PI's explicit instruction ("after the current batch
finishes and the 04d report is written"), launching it is the next
step, not yet taken. `puffin/fairness/attacker_in_the_loop.py`
(`refit_fresh_attacker_head`, `attacker_in_the_loop_train_step`; 3
tests pass), wired into `run_phase4d_approved_batch.py` via
`--mechanism attacker_in_the_loop --loss-variant {floor,group_dro}
--refit-every-k-epochs`, with wall-time logging. When launched: 3
datasets (GeoLife, Foursquare, **Gowalla-SF-Bay** -- note, different
Gowalla subset than 04d's `gowalla_nyc`, per the PI's explicit naming
for this formulation) x k in {1,5} x lambda4 in {0.3,1.0} x
loss_variant in {floor, group_dro} x 3 encoder seeds = 72 configs,
evaluated identically to 04d's mechanisms (Pareto fronts vs. plain
Mo-PAE and the no-privacy reference, residual-risk disparity,
leveling-down check), wall-time cost logged, then stop and report.

**PUFFIN-draft carryover landed in the paper** (committed, pushed,
test-compiled clean, prior to this round). Per the PI's instruction,
carried the earlier PUFFIN draft
(`/Users/mashhadi/Downloads/Puffin_FairMopae/main.tex`, local to the
PI's machine, not in this repo) into the new paper, rewritten in
double-blind third person, no numbers from that draft's own
experiments carried over: `sections/introduction.tex` (transportation-
equity motivation paragraph), `sections/related_work.tex` (new),
`sections/individual_fairness_formulation.tex` (new -- Eq. 1 restated
as a proper minimizable loss), `sections/put_base_model.tex` (new --
DRU/MPU/URU architecture, Definitions 1-3 redefined using the external
attacker), `main.bib` rewritten with VERIFIED/SOURCED annotations
(caught and fixed one real citation error: `zhan2022privacy` pointed to
an early arXiv preprint, corrected to the published PoPETs 2023 paper,
`zhan2023mopae`). The PI said "I'll add [the original draft's] LaTeX
source to paper/legacy/" -- not done by me; `paper/legacy/` does not
exist yet, left for the PI.

**Corrections applied earlier this round (all done, committed):**
1. **Decile-level reduction metrics** (`puffin/metrics/decile_reduction.py`,
   4 tests pass) replace the per-user-ratio `privacy_gain` in 02d
   onward: absolute reduction, relative reduction, and a new
   residual-risk-disparity metric (decile0/decile9 ratio), all
   decile-mean-based with bootstrap CIs, old per-user ratio moved to
   an appendix labeled unstable. `reports/02d_gowalla_filtered.md`
   rewritten with the corrected tables; its "gain lowest, sometimes
   negative, at decile 0" sentence corrected (relative reduction is
   consistently lowest at decile 0 in all 4 Gowalla subsets, not
   "sometimes negative" -- only Chicago's decile 0 CI touches zero).
2. **`floor_loss` inertness on GeoLife, diagnosed and fixed.**
   Confirmed via gradient-norm logging (`scripts/diagnose_floor_loss.py`)
   that the in-model adversary's confidence never exceeds ~0.008 there
   (chance 1/180=0.0056) because the encoder's own adversarial
   training suppresses it near chance by design -- the fixed
   `floor=0.3` default was ~54x unreachable, giving exactly zero
   gradient every step. Fixed: floor is now
   `floor_multiplier/total_users` (default 1x chance level), confirmed
   to give a genuinely nonzero gradient. Logged in
   `CHANGES_FROM_ORIGINAL.md`. GeoLife's floor_loss points are being
   rerun with the fix as part of the approved batch (see below).
3. **04c's "dominates" claims relabeled as trade-offs** (GeoLife and
   Foursquare) -- the PI's dominance definition (utility >= and risk <
   beyond both points' CIs) was never checked against those claims.
4. **04c now states explicitly** that every existing Phase 4 point
   used exactly 1 encoder-training seed (the 5-seed treatment is
   attacker-side only) -- ahead of the approved batch using >=3.

(Paper-draft status for every section, including the two filled in
this round, is covered in the "Current state" block above -- not
repeated here.)

## Superseded: new evaluation standard adopted, stopped after step 2 as instructed

The PI reviewed Checkpoint 3d and made the evaluation standard permanent:
**the primary attacker from here on is the hardened external attacker
(strongest per user across MLP, GRU, GBT) -- the adaptive attacker is
RETIRED as a primary standard.** 3d's decomposition is kept as a
robustness appendix: (b)-(a) relabeled "external side information about
regularity class," (e) showing group-aware attacks from released
features add no risk.

Next steps, in order, with an explicit "stop and report after step 2":
1. **Gowalla coverage fix** (`reports/02d_gowalla_filtered.md`) -- **DONE.**
2. **Phase 4 re-analysis on existing runs, no retraining**
   (`reports/04c_formulations_naive_standard.md`) -- **DONE.**
3. New training where step 2 shows gaps -- **NOT launched**, per "stop
   after step 2." Step 2 found exactly one real gap (named below);
   awaiting the PI's go-ahead before running it.
4. Correct earlier reports in place (dated "Superseded by 3d" notes on
   03/03b/03c/04) -- **NOT done yet**, also held pending the PI's
   go-ahead, out of caution about the literal "stop after step 2"
   instruction (this one is low-risk/already-decided bookkeeping, not
   new compute, so worth flagging explicitly in case the PI wants it
   done now regardless).

**Both required steps are complete. Stopping here, as instructed. No
paper drafting.**

**Headline findings:**
- Gowalla-NYC's coverage problem is NOT fixed by shortening SL (SL=5
  gives only 27.9% of users >=10 test windows, still well under the
  PI's ~50% bar) -- per the PI's own fallback, added gowalla_la/
  gowalla_chicago/gowalla_sf_bay as separate datasets instead of
  relaxing the filter. All four are now in the 24-37% pass-rate range
  (2235 filter-passing users combined) with real disparate-risk results:
  entropy survives the same regression controls in all four
  (p<1.8e-12 in the weakest case), same pattern as every earlier
  checkpoint. One real bug found and fixed along the way: Gowalla's
  user_id space is 4-5x sparser than its actual user count, which made
  the new standard's GBT component hang for 14+ hours (confirmed via
  py-spy it was genuinely executing, not deadlocked, just
  pathologically slow fitting a needlessly huge multiclass problem) --
  fixed by remapping to dense IDs before anything touches them.
- Phase 4's re-analysis hit a real blocking gap: **plain Mo-PAE's
  utility was never recorded in any existing run** (not Checkpoint 3b's
  driver, not any lambda4=0 Phase 4 run -- none exists). This blocks the
  leveling-down check completely (0 comparisons possible) and excludes
  `ed_soft` from the Pareto analysis entirely. Exactly 6 new training
  runs would close it (named in the report) -- not run, pending sign-off.
- Cross-formulation Pareto comparisons that don't need the missing
  plain-Mo-PAE reference: on Foursquare, `group_dro`/`per_user_minimax`
  dominate `floor`/`variance`/`group_adversarial` at comparable utility.
  GeoLife's `group_adversarial` at lambda4=0.3 has the best utility and
  a low max-min risk in that whole grid -- notable but unconfirmed
  against the still-missing plain-Mo-PAE reference.
- Cross-formulation (not vs. plain Mo-PAE, which the gap above blocks):
  on Foursquare, `group_dro`/`per_user_minimax` Pareto-dominate
  `floor`/`variance`/`group_adversarial` at comparable utility.
  GeoLife's `group_adversarial` at lambda4=0.3 has the best utility AND
  a low max-min risk in that whole grid -- notable, but unconfirmed
  against the missing plain-Mo-PAE reference.

**Stopping and reporting here, per "stop and report after step 2. No
paper drafting yet."**

## Superseded: Erebor infrastructure note (2026-10-20): the "restart" was a full OS/driver upgrade

What was announced as a restart turned out to be a full reinstall: erebor
is now Rocky Linux 10.2 (was an older release) with a new NVIDIA driver
(615.71.10, CUDA UMD 13.4). SSH's host key changed too (expected after a
reinstall, verified and accepted, not a security incident). This broke
the project's `.venv`: its python3.10 interpreter (`/usr/local/bin/python3.10`)
no longer exists -- only python3.12 is installed system-wide now.

**Fixed**: rebuilt `.venv` on python3.12. `torch==2.5.1+cu121` (the exact
same build as before) installs and runs correctly against the new
driver -- confirmed with a real CUDA matmul on all 4 GPUs, not just an
import check -- because PyTorch wheels bundle their own CUDA runtime and
don't depend on the system CUDA toolkit matching exactly; NVIDIA drivers
are backward-compatible with older CUDA runtimes, so the newer driver
runs the older-cu121 wheel fine. Every other pinned dependency
(numpy/pandas/pyarrow/scipy/scikit-learn/matplotlib/h3/etc.) reinstalled
at the exact same pinned version as before. **All 10 test files in
`tests/` pass cleanly on the rebuilt environment** -- ran every one, not
just a subset. `requirements.lock.txt` regenerated via `pip freeze`
(55 packages now vs. 68 before -- the old file appears to have been
stale/incomplete even before the upgrade, e.g. it never listed scipy or
scikit-learn despite both being installed and used; the new one is a
fresh, accurate `pip freeze`, not hand-edited).

**What this means for prior results**: nothing already committed is
invalidated -- Checkpoints 0 through Phase 4/5 ran under the old
Python-3.10 stack and their numbers stand as reported. Going forward,
everything runs under Python 3.12 with the identical pinned package
versions otherwise, so this is a toolchain footnote, not a correctness
concern, but worth knowing if anyone tries to reproduce an old run
bit-for-bit on a machine still running Python 3.10.

GPUs came back fully free after the reinstall (the unrelated
`mashhadi`-owned vLLM job occupying GPU 0 before the restart is gone).

## Current state (read this first): PI reframed, Phase 4 + 5 now running

The PI reviewed Checkpoint 3b and made a framing decision, superseding
3b's own "stop and report" call below (kept for history):
- **Not adopting** "naive privacy_gain measures nothing trustworthy" --
  that adversarially trained representations leak to post-hoc attackers
  is established prior work (Elazar & Goldberg 2018; Song & Shmatikov
  2020 -- both verified, see `reports/related_work_adversarial_leakage.md`),
  not a new finding of this project.
- **Adopted instead**: naive evaluation misstates privacy *unevenly
  across users*, so fairness-of-privacy claims built on it are
  unreliable. Checkpoint 3c (`reports/03c_uneven_gap.md`, pure analysis
  of already-collected 3b data, no new training) confirmed this: the
  naive-vs-adaptive measurement gap is significantly non-uniform across
  entropy deciles in all 6 panels tested (Kruskal-Wallis p<=0.017),
  consistently largest for the lowest-entropy (most vulnerable) users,
  and ranking users by naive privacy_gain disagrees substantially with
  ranking by adaptive privacy_gain (Spearman rho 0.16-0.67). Per the PI's
  continue condition, **proceeded straight to Phase 4.**

**Now running, both detached, in parallel** (see "Running now" below for
full detail):
1. **Phase 4 (item B)**: a grid evaluating 7 fairness formulations (the
   old `ed_soft`/L_pair_threshold baseline plus 5 new ones --
   `lipschitz_soft`, `floor_loss`, `variance_of_risk`, `group_dro_loss`,
   `per_user_minimax_loss` -- plus a genuinely new architecture, group-aware
   adversarial training) under the adaptive attacker, at lambda4=1.0,
   across all 3 datasets. All 7 formulations are implemented in
   `puffin/fairness/losses.py` and `puffin/fairness/group_adversarial.py`,
   tested (`tests/test_phase4_losses.py`, 9 checks, all pass).
2. **Phase 5 (item C) baselines**: the in-house baselines the PI asked
   for (plain Mo-PAE lambda4=0, variance regularizer, group-DRO, per-user
   minimax) are now fully covered for free by combining Checkpoint 3b's
   existing lambda4=0 data with the Phase 4 grid above -- no separate
   training needed. The remaining Phase 5 item, third-party TrajGAN and
   GI-DP (each time-boxed to half a day), was handed to a background
   general-purpose agent (matching this project's established pattern for
   POIPrivacy/MainTUL in Phase 3) -- see "Running now" for how to check on
   it.

## Superseded: Checkpoint 3b decision (kept for history)

Item A (harden the Task 5 adaptive-adversary result) is done — full
report: `reports/03b_adaptive_robustness.md`. 12 runs: 3 datasets
(GeoLife, Foursquare, Gowalla-NYC) x 4 lambda4 values (0, 0.3, 1.0, 3.0),
5 seeds each, bootstrap CIs.

**Finding: the specialized-attacker gain over a naive attacker is
statistically real (CI excludes zero) in all 12 runs, including every
plain-Mo-PAE (lambda4=0) run.** The group-reweighted variant shows the
same pattern on GeoLife and Foursquare (positive, overlapping-magnitude
gain even at lambda4=0), but reverses sign on Gowalla-NYC (negative under
plain Mo-PAE, positive once fairness is active) -- one genuine
fairness-specific exception out of six (3 datasets x 2 attack variants),
kept as a caveat in the report, not treated as overriding the dominant
pattern.

**Per the PI's explicit stop condition** ("stop and report if the gain
turns out not to be specific to the fairness term"), this is read as
"not specific" -- the dominant, cross-dataset, both-attack-variant
pattern is that any trained encoder is vulnerable to a subgroup-aware
attacker, fairness-active or not. **Stopping here. Not proceeding to
Phase 4 (B: new fairness-loss formulations) or Phase 5 (C: baselines)
construction** without the PI's review of this reframing. The report
recommends revising the paper's central question from "does Mo-PAE's
fairness mechanism survive an adaptive adversary" to the broader "does
naive-attacker privacy_gain measure anything trustworthy, for any
mechanism" -- which would change what Phase 4/5 need to compare if the PI
decides to continue.

Also newly answered: a user's entropy decile is itself recoverable from
released encoder features alone (34-42% top-1 vs. 10% chance, all three
datasets) -- the adaptive adversary's side-information requirement is
realistic, not a hypothetical worst case (item A.4, the "state the threat
model" ask).

## Superseded: PI correction after Checkpoint 3 (kept for history)

Two corrections to how Checkpoint 3 was framed:
1. **Phase 4 is not skipped.** The earlier "proceed to Phase 5" instruction
   was a mistake on the PI's part — Phase 4 (the paper's method
   contribution: L_pair_threshold/ed, L_lipschitz, L_floor, plus variance-
   of-gain and group-DRO regularizers) stays in, and can run IN PARALLEL
   with Phase 3/5, not after them.
2. **The Task 5 adaptive-adversary result DOES change the framing.** It
   isn't just "a strengthening detail" — it reframes the paper's central
   method question from "does Mo-PAE+ed_soft achieve individual fairness"
   to **"does ANY fairness formulation give protection that survives an
   adaptive adversary?"** Every Phase 4 formulation must now be judged by
   that standard, at matched utility, not just by the naive-attacker
   privacy_gain metric Checkpoints 2/2b/2c used throughout.

**Gating step before anything else** (PI's explicit stop condition): does
the adaptive-adversary gain found in Checkpoint 3 (2.4-2.8x on the
fairness-active model's most-protected decile) ALSO appear against PLAIN
Mo-PAE (lambda4=0)? If yes — the effect is generic to any trained model,
not specific to the fairness mechanism — **stop and report, do not proceed
to Phase 4/5 construction.** If no — the fairness term itself is what
creates the exploitable gap — continue to the full hardening (3b), then
Phase 4 (judged by the adaptive-adversary standard) and Phase 5 in
parallel.

**Single-seed gating check already run** (`geolife_lambda4_0`, see old
"Running now" below, superseded): multipliers 2.31x/3.07x vs. Checkpoint
3's 2.37x/2.83x at lambda4=1.0 — strikingly similar, leaning toward "not
specific to the fairness term," but one seed each isn't enough to decide
something this consequential. The rigorous version (5 seeds + bootstrap
CIs, full lambda4 grid, all 3 datasets) is running now; see "Running now"
below for how to read it when it's done.

## Checkpoint 2c, in 5 lines
(a) Yes — entropy survives the user-clustered regression on all three datasets (e.g. GeoLife S_unc coef -0.044, 95% CI [-0.065, -0.023], excludes zero; Foursquare/Gowalla-NYC likewise).
(b) The low- vs high-entropy risk gap holds under BOTH splits, on all three datasets — if anything more cleanly under the contemporaneous split (e.g. GeoLife decile 0 = 81.7% vs decile 9 = 21.0%, non-overlapping CIs) than under the future split.
(c) Strongest defensible claim: plain Mo-PAE reliably protects middle-entropy users but gives the lowest- and highest-entropy users no statistically detectable benefit — it does not equalize risk, and the people most at risk (lowest entropy) see no significant gain.
(d) Retracted: 2b's claim that GeoLife's two lowest-entropy deciles get actively WORSE — under seed-paired bootstrap CIs decile 0 includes zero and decile 1 flips sign; the point estimates in 2b came from comparing two independently-averaged 5-seed means rather than seed-paired differences, which overstated how reliable that specific apparent harm was.

## Phase 3 task 4, in 5 lines
(a) Unlike task 3's POIPrivacy, MainTUL (https://github.com/Onedean/MainTUL) actually ran: after two trivial compat fixes (numpy's removed `np.Inf`; a logging format-string bug that was broken even under the authors' own default args), its own `foursquare_mini` demo trained and tested end to end.
(b) Because it ran, ported (not vendored-imported) its mutual-distillation LSTM-student/Transformer-teacher architecture into `puffin/attacks/tul_maintul.py`, dropping only the POI-category view (GeoLife has no POI category data, same reasoning as task 3's POIPrivacy writeup).
(c) Validated on GeoLife's own real trajectories (163 non-disjoint users, 13,611 train / 3,403 test trajectories): student (the network actually meant for deployment) gets overall top-1 0.557, top-5 0.795 on held-out trajectories — chance is ~1/163 ≈ 0.006.
(d) That pooled number is trajectory-weighted and dominated by a handful of very prolific users (one user with 1,722 train trajectories hits 0.79 top-1 alone); the median PER-USER top-1 is only 0.071 — most users have just a handful of test trajectories and are much harder to link than the pooled number suggests, a disparate-risk pattern consistent with Phase 2's findings, now worth carrying into TUL once a real TrajGAN/GI-DP target exists.
(e) Tested: `tests/test_tul_maintul.py` (4 checks, all pass). Full account: `CHANGES_FROM_ORIGINAL.md`.

## Phase 3 task 5 (last), in 5 lines — full checkpoint: `reports/03_attack_suite.md`
(a) Trained an actually fairness-ACTIVE Mo-PAE (`ed_soft`, `lambda4=1.0`) on GeoLife — every prior Mo-PAE run in this project (Checkpoints 2/2b/2c) kept `lambda4=0` ("plain Mo-PAE") on purpose; this is the first one testing the fairness mechanism itself.
(b) Found its most-protected entropy decile (5) via the ordinary group-agnostic attacker; checked it wasn't a near-zero-denominator artifact (unlike one other decile's -4.69 mean gain, which was) before trusting it.
(c) A fairness-aware adaptive adversary who knows that group recovers 2.4x (per-group specialized) to 2.8x (group-reweighted) more risk than the naive attacker on exactly that group (18%→31–37% top-1); the reweighted variant's effect is cleanly concentrated there (~0 change for everyone else), exactly as designed.
(d) This answers the plan's "old paper's own limitation" directly: a mechanism's apparent fairness, measured against a naive attacker, can be partly an artifact of not testing against the right adversary.
(e) Doesn't change the framing — strengthens the existing story — so proceeding straight to Phase 5 per the "proceed unless" rule.

A fresh session should be able to resume from this file alone. Repo:
`/data/mashhadi/puffin2` on `erebor.uwb.edu` (ssh `mashhadi@erebor.uwb.edu`),
pushed to `https://github.com/ajahanba/puffin2` (private). Plan:
`puffin_v2_plan.md` (kept outside this repo, with the PI). Full account of
every deviation/bugfix: `CHANGES_FROM_ORIGINAL.md`.

**Environment:** `.venv` (Python 3.10, PyTorch 2.5.1+cu121, 4x A4500 GPUs),
pinned in `requirements.lock.txt`. `/home` on erebor is nearly full (not our
problem); everything lives under `/data/mashhadi/puffin2`.

## Erebor reliability — two unexplained reboots, for the PI to raise with the sysadmin

Confirmed via `last reboot`/`who -b` (both "still running" until the next
one superseded them):
- **2026-10-16, ~17:20** (local erebor time) — discovered mid-session when
  an SSH connection was refused; on reconnect `uptime` showed 0 minutes.
  Corrupted two Python packages mid-write (matplotlib, contourpy — both
  left as 0-byte `__init__.py`); both reinstalled and verified.
- **2026-10-17, ~12:01** — discovered at the start of a later session via
  `uptime`. No corruption found this time (nothing was mid-write).

**Cause undetermined.** Checked and ruled out: `dnf-automatic` is not
installed or configured (no `/etc/dnf/automatic.conf`, no
`dnf-automatic(-install).timer` unit) — these are NOT unattended-upgrade
reboots. No `at` jobs, no user crontab, and `/etc/cron.d/` has only the
standard `0hourly`/`raid-check` entries (neither reboots anything).
`/var/log/messages` (and its rotated copies) would likely show the actual
cause but are root-only; `journalctl --list-boots` only retains the
current boot, so the in-session logs for either reboot are gone. Could be
sysadmin-initiated maintenance, a host-level/power event, or something else
infrastructure-side — worth asking directly since it's now happened twice
in two days.

## Operating policy going forward (per the PI)

Every long-running job (attacker training, sweeps) runs detached
(`nohup ... &` with `disown`, as already practiced) and writes results
incrementally rather than only at the very end, so a reboot loses at most
the in-flight run, not completed work — true so far since every checkpoint
has committed its outputs as soon as a script finishes, but worth stating
as a hard rule now that reboots are a real, recurring risk, not a one-off.
Every session ends by launching the next batch of compute jobs (not idle)
and recording here what's running and how to check it (see "Running now"
below, kept current) — so a reboot or a fresh session never has to guess
what was in flight.

## Running now

**Checkpoint 04d approved batch, in progress (the blocking dependency
for the final rebuild-and-stop step).**
`scripts/launch_phase4d_approved.sh`, launched detached ~11:12 AM PDT
2026-10-21, capped at 2 concurrent GPUs (0, 1 -- GPUs 2/3 left free per
the PI), `nice -n 10`, `OMP_NUM_THREADS`/`MKL_NUM_THREADS` capped at 16.
21 runs: `plain` + `no_privacy` + `ed_soft` (lambda4 in {0,0.3,1.0,3.0})
per dataset (3 datasets), plus GeoLife's fixed `floor` rerun (lambda4
in {0.3,1.0,3.0}). Each run uses >=3 encoder-training seeds x 5 attacker
seeds, the new strongest-per-user (MLP+GRU+GBT) standard -- substantially
heavier than any earlier grid (3x the encoder training, plus GBT fit on
every one of up to 3 encoder seeds' frozen output), so expect this to
take considerably longer than previous grids (hours, not tens of
minutes) -- check actual progress before assuming it's stuck.

- Logs: `logs/phase4d/<dataset>_<mechanism>_lambda<l4>.log`, orchestrator
  `logs/phase4d/orchestrator.log`.
- Output: `reports/figures/phase4d/<dataset>_<mechanism>_lambda<l4>/{summary.json,per_user_04d.csv}`.
- **If resuming cold:** check `tail logs/phase4d/orchestrator.log` and
  `ls reports/figures/phase4d/*/summary.json | wc -l` (target: 21). If
  the orchestrator is gone but not all 21 exist, relaunch
  `nohup bash scripts/launch_phase4d_approved.sh > logs/phase4d/orchestrator_stdout2.log 2>&1 < /dev/null & disown`
  -- skips completed runs. Re-check `nvidia-smi` for a free GPU pair
  first.
- **Once all 21 finish, the remaining work (not started yet):**
  1. Apply `puffin.metrics.decile_reduction` to each `plain`/`no_privacy`
     run's per-user output -- 04d's own driver only computed the OLD
     `bootstrap_decile_ci`, not the new decile-mean reduction metrics,
     so this is a separate quick analysis pass, not already done.
  2. Rebuild `reports/04c_formulations_naive_standard.md`'s Pareto
     fronts with plain Mo-PAE and the no-privacy reference overlaid as
     the utility axis's two reference points (now that both have a
     logged utility).
  3. Run the leveling-down check (blocked before on missing plain-Mo-PAE
     utility; should now be computable).
  4. Fill in `paper/sections/residual_risk_disparity.tex` (part 2) using
     the result of step 1, and `paper/sections/phase4_outline.tex`'s
     real content using steps 2-3 -- regenerate tables via
     `paper/scripts/generate_figures.py`, do not hand-write.
  5. **Stop and report, per the PI's instruction** -- do not proceed to
     step 4's "correct earlier reports in place" (dated superseded
     notes on reports/03, 03b, 03c, 04) or further paper sections
     without checking in first, since that was scoped as a separate,
     later ask.

**Phase 5b restarted at low priority** (`scripts/launch_phase5b_grid.sh`,
~11:15 AM PDT 2026-10-21): `nice -n 19`, threads capped to 6,
batch_size reduced from 4 to 2 -- the original uncapped 4-at-a-time run
drove load average to ~79-110 and made zero progress in 14+ hours
before being killed in a previous session. Needed for the paper's
baselines per the PI; genuinely low priority, no urgency either way.

**Known infrastructure quirk this session**: erebor went fully
unreachable (SSH connection timeout, not a crash -- same boot time
both times) twice under heavy combined GPU+CPU load. Recovered on its
own both times once load dropped. Keeping concurrent job count modest
(2 GPUs, capped threads, `nice`) going forward for this reason, per the
PI's own instruction.

Old single-seed gating check (`geolife_lambda4_0`,
`reports/figures/phase3_adaptive/geolife_lambda4_0/`) is superseded by
the grid's `geolife_lambda4_0.0` run; kept on disk for reference, not
deleted.

Old single-seed gating check (`geolife_lambda4_0`,
`reports/figures/phase3_adaptive/geolife_lambda4_0/`) is superseded by
the grid's `geolife_lambda4_0.0` run; kept on disk for reference, not
deleted.

## What's done

- **Checkpoint 0** (`reports/00_mopae_audit.md`, `00b_pets_provenance.md`):
  audited `mopae.py` against the real PoPETs 2023 paper and Utkarsh's PUFFIN
  fork. Every structural issue in the audit table is live in the fork
  except #5. Found `ed`-loss is gradient-free in the original (inert).
  Decision: do not present old PUFFIN numbers, even as a pilot.
- **Phase 1** (data layer): 9 built datasets under `data/processed/` —
  `geolife` (raw Microsoft 1.3, 181 users, 24.9M records — the main GeoLife
  dataset now), `geolife_sample` (Mo-PAE's small sample, Phase 0
  legacy-mode comparisons only), `foursquare` (1083 users), `gowalla_us`
  (scalability study only), `gowalla_{nyc,la,chicago,sf_bay}` (metro
  subsets, the main Gowalla unit), `yjmob100k_full` (100,000 users, 111.5M
  rows). Shared infra: `puffin/data/{schema,tessellation,splits,filtering,
  stats}.py`. Two serious bugs found+fixed along the way: an index-
  misalignment bug (silently NaN-filled ~32% of Gowalla) and a quadratic
  bug in `membership_split` (stuck 90+ min on YJMob100K before being
  killed; fixed, reruns in 4m09s) — both have regression tests in `tests/`.
- **Checkpoint 2 / 2b** (`reports/02_disparate_risk.md`,
  `02b_disparate_risk_v2.md`): disparate re-id risk confirmed on GeoLife,
  Foursquare NYC, and Gowalla-NYC (all after a PI-directed windowing fix —
  check-in data windows over each user's full sequence, not daily
  pseudo-trajectories; Foursquare went from 71/1083 usable users to
  1083/1083). Hardened attacker: best of {MLP, GRU} per seed, 5 seeds,
  early stopping. **Key finding: plain Mo-PAE (lambda4=0) under-protects
  low-entropy users on all three datasets** — GeoLife's two lowest-entropy
  deciles get worse, not just less-helped. Per the PI's conditional, this
  does NOT stop the project — proceeding to Phase 3.
  Code: `puffin/attacks/{windowing,reid,analysis}.py`,
  `scripts/run_phase2_disparate_risk.py`.

- **Checkpoint 2c** (`reports/02c_robustness.md`): robustness checks on 2b,
  all three datasets rerun with seed-paired bootstrap CIs, cluster-robust
  regression, chronological early stopping, and a new contemporaneous-vs-
  future split comparison. **Retracted one claim from 2b**: GeoLife's two
  lowest-entropy deciles do NOT robustly get worse under Mo-PAE (CI
  includes zero) — the more defensible claim is "no significant protection
  either way" for the lowest/highest deciles, with only middle deciles (3-6)
  robustly improving. Foursquare/Gowalla-NYC's "everyone improves in
  absolute terms" from 2b held up fine under the same scrutiny. New finding:
  drift (contemporaneous accuracy minus future accuracy) varies by entropy
  decile on all three datasets, but in OPPOSITE directions for GPS
  (GeoLife: low-entropy users drift most) vs. check-in data (Foursquare/
  Gowalla-NYC: low-entropy users drift least) — worth keeping for Phase 8.
  Top-level conclusion unchanged: proceeding to Phase 3 (stop condition not
  triggered).
  Code: `puffin/attacks/windowing.py`'s `build_sl_windows_contemporaneous`
  and `_chronological_val_split`/per-seed-preserving changes in `reid.py`,
  `ols_cluster_robust` in `analysis.py`,
  `scripts/run_phase2c_robustness.py`.

## Phase 3, in this PI-specified order

1. **Done.** Threat-model table (`puffin/attacks/README.md`) + metrics
   module (`puffin/metrics/{privacy,fairness,leveling_down}.py`:
   privacy_gain, protection_at_k, similar_pairs_within_epsilon,
   lipschitz_violation, max_min_gain, gini, gain_percentiles,
   leveling_down_check; `puffin/attacks/analysis.py` gained
   `paired_bootstrap_test`). Tested: `tests/test_metrics.py`.
2. **Done.** Finalized external re-id attacker:
   `train_raw_attacker_strongest_per_user` in `puffin/attacks/reid.py` —
   MLP + GRU + gradient-boosted trees (sklearn `HistGradientBoostingClassifier`),
   per-user element-wise max (not one global per-seed winner, per the
   plan's literal wording), each user's row records which architecture won.
   2b/2c's `train_raw_attacker_hardened` (2-arch, per-seed winner) is left
   unchanged so those checkpoints stay reproducible. Verified on real
   GeoLife data: `scripts/smoke_test_phase3_attacker.py`.
3. **Done.** Membership inference. POIPrivacy
   (https://github.com/KunlinChoi/POIPrivacy) given a genuine, bounded
   attempt (cloned, read in full, its own demo run successfully) but ruled
   out for two independent structural reasons: its LocMIA/TrajMIA attacks
   are hardwired to GETNext, a specific POI-recommender architecture, whose
   dense POI-adjacency graph would need ~29GB for GeoLife's 85,074 cells,
   and whose required POI-category node features don't exist in this
   project's schema. Built in-house instead:
   `puffin/attacks/mia_inhouse.py`, a classic Yeom/Shokri loss-threshold
   MIA attacking Mo-PAE-fixed's own next-location model, with a
   calibration split (avoids circularity) and per-user output. Real test on
   GeoLife: overall AUC 0.555, accuracy 0.598 (181 users, SL=10) — a real,
   modest, non-suspicious signal. Tested: `tests/test_mia_inhouse.py` (5
   checks, all pass). Full reasoning logged in `CHANGES_FROM_ORIGINAL.md`.
4. **Done.** Trajectory-user linking. MainTUL
   (https://github.com/Onedean/MainTUL) given the same genuine, bounded
   attempt as task 3's POIPrivacy -- cloned to `legacy/MainTUL/` (not
   committed), read in full, two trivial compat bugs fixed (`np.Inf`
   removed in numpy>=2.0; a logging format-string mismatch that crashed
   the authors' own default training config on its very first logged
   batch), and its `foursquare_mini` demo trained+tested successfully end
   to end. Because it ran (unlike POIPrivacy), ported its mutual-
   distillation LSTM-student/Transformer-teacher method into
   `puffin/attacks/tul_maintul.py` instead of importing the uncommitted
   clone, dropping only the POI-category view (no POI category data in
   this project's schema, same reasoning as task 3). Validated on real
   GeoLife trajectories (163 non-disjoint users): overall top-1 0.557 /
   top-5 0.795 (student), chance ~0.006 -- a real, large, non-suspicious
   signal, with a skewed per-user distribution (median per-user top-1 only
   0.071; a few very-prolific users drive the pooled number). Only applies
   to mechanisms that release trajectories (TrajGAN, GI-DP), neither of
   which Phase 5 has built yet, so this was validated against GeoLife's
   own trajectories as a stand-in released-trajectory set, not a real
   attack on a trained baseline. Tested: `tests/test_tul_maintul.py` (4
   checks, all pass). Full reasoning logged in `CHANGES_FROM_ORIGINAL.md`.
5. **Done.** Fairness-aware adaptive adversary (`puffin/attacks/adaptive.py`:
   `per_group_specialized_attack`, `group_reweighted_attack`;
   `scripts/run_phase3_adaptive_adversary.py` trains the first
   fairness-ACTIVE Mo-PAE in this project and attacks it). Result: a 2.4-2.8x
   risk recovery on the mechanism's most-protected decile vs. a naive
   attacker — see `reports/03_attack_suite.md` for the full numbers and the
   near-zero-denominator sanity check. Tested: `tests/test_adaptive.py`
   (4 checks, including a planted-signal case). **Phase 3 is now fully
   done.**

**Time-box rule** (already applied once, for task 3): if a third-party repo
doesn't run on our data after ~half a day of real effort, stop, implement
an in-house equivalent, document why in `CHANGES_FROM_ORIGINAL.md`, move on.

Phase 3 checkpoint report written: `reports/03_attack_suite.md` (5-line
summary at top). Continuing straight to Phase 5 baselines — this result
doesn't change the
framing (same "proceed unless" rule as Checkpoint 2b/2c).

## Open questions / things to watch

- Gowalla-NYC's SL=10 test coverage is only 587/1517 (39%); SL=5 recovers
  1154/1517. Not yet a problem, but worth revisiting if a later phase needs
  fuller coverage.
- `lambda_priv` was swept on an arbitrary grid (0.05/0.1/0.3); not a tuned
  baseline, just enough to see the pattern. Phase 5's baseline sweep should
  not treat these 3 points as final.
- Checkpoint 2c's drift-by-decile finding (opposite direction for GPS vs.
  check-in data) has no explanation yet, just an observation across 3
  datasets/1 seed-set each — worth a sanity check if it becomes load-bearing
  for any later claim.
