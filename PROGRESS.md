# PUFFIN v2 — Progress

## Checkpoint 2c, in 5 lines
(a) Yes — entropy survives the user-clustered regression on all three datasets (e.g. GeoLife S_unc coef -0.044, 95% CI [-0.065, -0.023], excludes zero; Foursquare/Gowalla-NYC likewise).
(b) The low- vs high-entropy risk gap holds under BOTH splits, on all three datasets — if anything more cleanly under the contemporaneous split (e.g. GeoLife decile 0 = 81.7% vs decile 9 = 21.0%, non-overlapping CIs) than under the future split.
(c) Strongest defensible claim: plain Mo-PAE reliably protects middle-entropy users but gives the lowest- and highest-entropy users no statistically detectable benefit — it does not equalize risk, and the people most at risk (lowest entropy) see no significant gain.
(d) Retracted: 2b's claim that GeoLife's two lowest-entropy deciles get actively WORSE — under seed-paired bootstrap CIs decile 0 includes zero and decile 1 flips sign; the point estimates in 2b came from comparing two independently-averaged 5-seed means rather than seed-paired differences, which overstated how reliable that specific apparent harm was.

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

A background agent is investigating Phase 3 task 3 (membership inference
via POIPrivacy, https://github.com/KunlinChoi/POIPrivacy) on erebor: clone,
read, attempt a real integration against GeoLife within a time-box; if it
doesn't run after a genuine bounded effort, fall back to an in-house
loss-threshold MIA (`puffin/attacks/mia_inhouse.py`) and document why in
`CHANGES_FROM_ORIGINAL.md`. No way to check progress directly (it's a
different Claude session, not a shell job) — its outcome will arrive as a
report; if resuming cold, just check `git log` on `erebor:/data/mashhadi/puffin2`
for a commit mentioning MIA/POIPrivacy to see whether it already finished,
and check `puffin/attacks/` for `mia_poiprivacy.py` or `mia_inhouse.py`.

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

## Next concrete action — Phase 3, in this PI-specified order

1. Threat-model table + metrics module (Phase 3.1/3.3 of the plan).
2. Fresh external re-id attacker, finalized on top of what exists
   (`puffin/attacks/reid.py` already has a hardened one from 2b/2c — reuse,
   don't rebuild).
3. Membership inference via POIPrivacy (third-party repo).
4. TUL via MainTUL (third-party repo) — only against mechanisms that
   release trajectories (TrajGAN, GI-DP), not Mo-PAE/PUFFIN directly.
5. Adaptive adversary, last.

Nothing on this list is started yet.

**Time-box rule:** if MainTUL or POIPrivacy doesn't run on our data after
~half a day of real effort each, stop, implement an in-house equivalent
(GRU-based TULER-style TUL; loss-threshold MIA), document why in
`CHANGES_FROM_ORIGINAL.md`, and move on. Don't sink more time than that
into third-party integration.

Write a Phase 3 checkpoint report (5-line summary at top) when done, then
continue straight to Phase 5 baselines unless something changes the
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
