# PUFFIN v2 — Progress

A fresh session should be able to resume from this file alone. Repo:
`/data/mashhadi/puffin2` on `erebor.uwb.edu` (ssh `mashhadi@erebor.uwb.edu`),
pushed to `https://github.com/ajahanba/puffin2` (private). Plan:
`puffin_v2_plan.md` (kept outside this repo, with the PI). Full account of
every deviation/bugfix: `CHANGES_FROM_ORIGINAL.md`.

**Environment:** `.venv` (Python 3.10, PyTorch 2.5.1+cu121, 4x A4500 GPUs),
pinned in `requirements.lock.txt`. `/home` on erebor is nearly full (not our
problem); everything lives under `/data/mashhadi/puffin2`.

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

## In progress right now

**Checkpoint 2c (robustness checks the PI asked for before Phase 3 goes
deep), writing `reports/02c_robustness.md`:**
1. GeoLife negative-risk-reduction: pair raw/post by seed (not by
   seed-averaged means), bootstrap 95% CI over users per decile, state
   plainly whether the low deciles' CI excludes zero.
2. Regression: user-clustered SEs (or user-level), report coefficients +
   CIs not just p-values.
3. Attacker gap: redo early stopping with a chronological (not random)
   per-user validation split; compare disparate risk under (a) a
   contemporaneous split (no time gap between attack-model training data
   and target data) vs (b) the existing future/temporal split; check
   whether the entropy gap holds under (a); check whether drift itself
   (future-accuracy minus contemporaneous-accuracy, per user) varies by
   entropy decile.

**Not started yet:** none of this is written yet. Next concrete action:
implement chronological-val-split + per-seed-preserving attacker functions
in `puffin/attacks/reid.py`, a contemporaneous-split windower in
`windowing.py`, and `ols_cluster_robust`/paired-bootstrap in `analysis.py`.

## Then, in this order (PI-specified)

1. Phase 3.1 threat-model table + Phase 3.3 metrics module (as planned).
2. Fresh external re-id attacker, finalized on top of what exists.
3. Membership inference via POIPrivacy (third-party repo).
4. TUL via MainTUL (third-party repo) — only against mechanisms that
   release trajectories (TrajGAN, GI-DP), not Mo-PAE/PUFFIN directly.
5. Adaptive adversary, last.

**Time-box rule:** if MainTUL or POIPrivacy doesn't run on our data after
~half a day of real effort each, stop, implement an in-house equivalent
(GRU-based TULER-style TUL; loss-threshold MIA), document why in
`CHANGES_FROM_ORIGINAL.md`, and move on. Don't sink more time than that
into third-party integration.

Write a Phase 3 checkpoint report (5-line summary at top) when done, then
continue straight to Phase 5 baselines unless something changes the
framing (same "proceed unless" rule as Checkpoint 2b).

## Open questions / things to watch

- Gowalla-NYC's SL=10 test coverage is only 587/1517 (39%); SL=5 recovers
  1154/1517. Not yet a problem, but worth revisiting if a later phase needs
  fuller coverage.
- The attacker val/test generalization gap (51-98% val vs 17-57% test,
  consistent across all 3 datasets) might be pure temporal drift or might
  be partly an artifact of validation being a random (non-chronological)
  subsample of train — this is exactly what Checkpoint 2c's item 3 is
  checking.
- `lambda_priv` was swept on an arbitrary grid (0.05/0.1/0.3); not a tuned
  baseline, just enough to see the pattern. Phase 5's baseline sweep should
  not treat these 3 points as final.
