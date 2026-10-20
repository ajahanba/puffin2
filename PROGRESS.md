# PUFFIN v2 — Progress

## Erebor infrastructure note (2026-10-20): the "restart" was a full OS/driver upgrade

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

**Erebor is about to be restarted (PI-initiated, 2026-10-19 evening) --
everything below is committed and pushed (`9140e09`), nothing was at risk.**

**1. Phase 4 formulation grid: all 21 runs DONE**, committed in `9140e09`.
7 formulations (`ed_soft`, `lipschitz`, `floor`, `variance`, `group_dro`,
`per_user_minimax`, `group_adversarial`) x 3 datasets, lambda4=1.0.
Analysis so far (`scripts/summarize_phase4.py`, `scripts/analyze_phase4_gaps.py`,
output in `reports/figures/phase4/{combined_summary,gap_ci_summary}.json`):

- **Headline finding**: `group_adversarial` is the ONLY formulation whose
  adaptive-minus-naive gap bootstrap CI includes zero or goes negative,
  on all 3 datasets. Every other formulation, including the old PUFFIN
  `ed_soft` baseline, shows a reliably POSITIVE gap (CI excludes zero)
  on all 3 datasets -- i.e. changing the loss function alone never closes
  the gap; only changing what the in-model adversary IS during training
  does.
- **But this is utility-confounded on 2 of 3 datasets.** On GeoLife,
  `group_adversarial`'s utility (0.660) is close to `ed_soft`'s (0.680)
  -- a genuine, roughly matched-utility comparison. On Foursquare and
  Gowalla-NYC, `group_adversarial`'s utility at lambda4=1.0 collapses
  (0.043, 0.050) far below every other formulation (0.055-0.288) --
  so its near-zero/negative gap there might just be "a badly broken
  encoder is hard for any attacker," not genuine fairness-specific
  robustness.
- **Followed up with group_adversarial at lambda4=0.3 on Foursquare**
  (`reports/figures/phase4/foursquare_group_adversarial_lambda0.3/`,
  committed): utility only partially recovered (0.088, still well below
  other formulations) and the gap REAPPEARED clearly positive (+0.111,
  CI [0.078,0.143]) -- consistent with the utility-collapse-artifact
  explanation, not genuine robustness, on Foursquare specifically.
- **The equivalent Gowalla-NYC follow-up (lambda4=0.3) never actually
  launched** -- queued in the same multi-line ssh command as the
  Foursquare one, which is error-prone in this environment (apostrophe/
  quoting issues have bitten this project before); no log file was ever
  created, no process ran. Not critical (GeoLife's clean result and
  Foursquare's debunked one already tell most of the story), but worth
  doing before finalizing `reports/04_formulations.md`: rerun as its OWN
  single ssh command, not bundled with another launch:
  ```
  ssh mashhadi@erebor.uwb.edu "cd /data/mashhadi/puffin2 && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=2 nohup python scripts/run_phase4_formulations.py --dataset gowalla_nyc --parquet data/processed/gowalla_nyc/unified.parquet --out-dir reports/figures/phase4/gowalla_nyc_group_adversarial_lambda0.3 --formulation group_adversarial --no-respect-traj-boundary --sl 10 --max-windows-per-user 200 --mopae-epochs 20 --lambda-priv 0.3 --lambda4 0.3 --attack-epochs 25 --global-seeds 0,1,2,3,4 --adaptive-seeds 0,1,2,3,4 --reweight-factor 10.0 --device cuda > logs/phase4/gowalla_nyc_group_adversarial_lambda0.3.log 2>&1 < /dev/null & disown"
  ```
  (check `nvidia-smi` first for a free GPU -- GPU 0 was occupied by an
  unrelated `mashhadi`-owned vLLM job before the restart; may or may not
  still be, check fresh).
- **Also noted**: in every (dataset, formulation) pair checked, the
  decile the NAIVE attacker calls "most protected" is never the same
  decile that's actually worst-off under the ADAPTIVE attacker -- the
  same ranking-disagreement pattern Checkpoint 3c found, showing up again
  at the individual-formulation level.

**Next step**: get the Gowalla-NYC lambda4=0.3 follow-up (above), then
write `reports/04_formulations.md` (5-line summary at top) covering all
of the above honestly, including the utility-confound caveat -- do not
oversell `group_adversarial` as a clean win without that caveat front
and center.

**2. Phase 5 (item C) third-party baselines: DONE**, committed and pushed
(`ec07f4b`, `8859366`). TrajGAN (LSTM-TrajGAN) ported and validated on
real GeoLife (`puffin/baselines/trajgan.py`). GI-DP has no canonical repo
-- built in-house from the Planar Laplace mechanism's own published
formula, validated against its closed-form mean (`puffin/baselines/gi_dp.py`).
Full writeup: `reports/05_baselines_trajgan_gidp.md`. Neither has been
attacked with TUL/re-id/MIA yet -- that's this project's next step for
them, not done here. In-house Phase 5 baselines (plain Mo-PAE, variance
regularizer, group-DRO, per-user minimax) are already fully covered by
Checkpoint 3b's lambda4=0 data plus the Phase 4 grid above -- no separate
training was needed for those.

**If resuming cold after the restart:** nothing is running. Start with
the Gowalla-NYC follow-up above, then write `reports/04_formulations.md`.

**2. Phase 5 (item C) third-party baselines -- DONE, both committed and
pushed** (`ec07f4b`): TrajGAN (LSTM-TrajGAN, Rao et al. 2021) ran
essentially out of the box -- the cleanest third-party outcome in this
project so far, cleaner even than MainTUL -- after one trivial dependency
pin (`h5py==2.10.0`, a Keras-2.2.4/h5py>=3.0 API mismatch in the weight
loader, hit at generation time not training time). Ported into
`puffin/baselines/trajgan.py`: PyTorch, this project's schema, venue
`category` dropped (no such field here, same reasoning as MainTUL's
POI-category drop), synthetic `cell_id` added via re-tessellation (the
GAN itself only emits continuous lat/lon), `max_traj_len=150` truncation
(GeoLife trajectories run up to 92,645 points), and a robust
99.5th-percentile `scale_factor` in place of the original's true max --
found necessary because GeoLife's raw GPS data has physically-impossible
outlier fixes (lat up to ~400 degrees) that would otherwise dominate
normalization. GI-DP turned out to have **no canonical repo at all**
(checked GeoPrivacy -- sporadic/single-point only -- and privkit -- a
different adaptive variant -- neither fits "the classical mechanism"), so
per the task's own license to use judgment, built in-house directly from
Andres et al. (2013)'s own Planar Laplace formula:
`puffin/baselines/gi_dp.py`, verified against the distribution's own
closed-form `E[r]=2/epsilon` (matched to within 0.2% on synthetic data,
near-exact on real GeoLife). Both tested (`tests/test_trajgan.py`,
`tests/test_gi_dp.py`, all checks pass) and validated on real GeoLife
(`scripts/smoke_test_phase5_{trajgan,gidp}.py`) -- TrajGAN: 30 epochs,
19m14s CPU wall time, G loss 24,875->678, lat/lon reconstruction error
148.6/144.8->9.7/16.5 degrees (epoch 2 -> epoch 30); GI-DP: ran across
eps in {0.001,0.01,0.1} in ~110s each, mean perturbation distance
matched theory almost exactly at every epsilon. Full account:
`CHANGES_FROM_ORIGINAL.md`; checkpoint report:
`reports/05_baselines_trajgan_gidp.md`. **Not done** (explicitly out of
this task's scope, left for a later phase): running TUL/re-id/MIA
against either mechanism's output -- both now produce real schema-valid
synthetic/perturbed data on GeoLife for that evaluation to target.

**If resuming cold:** read this section, then check
`logs/phase4/orchestrator.log` and whether `reports/04_formulations.md`
or `reports/05_baselines_trajgan_gidp.md` exist yet.

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
