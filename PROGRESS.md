# PUFFIN v2 — Progress

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

Nothing compute-heavy in flight. **Phase 3 is fully done** (all 5 tasks;
`reports/03_attack_suite.md`). Starting Phase 5 (baselines) next, per the
PI's explicit "continue to Phase 5 baselines" (skipping Phase 4 — the PI's
own instruction, not an oversight; Phase 4's core piece, the `ed_soft`
fairness loss, already exists from the Checkpoint 0 fairness-loss port).

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
