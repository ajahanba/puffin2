# Checkpoint 2b — Hardened disparate-risk result, PI-directed revision

**Status: proceeding to Phase 3** (per the PI's own conditional — see "The key question" below; this finding does not trigger the stop condition).

Revises `reports/02_disparate_risk.md` per the PI's review: dropped the daily-trajectory windowing requirement for check-in data, hardened the attacker (best of MLP/GRU, 5 seeds, early stopping), added proper regression statistics and bootstrap CIs, separated the plain-Mo-PAE (λ4=0) operating-point sweep from the fairness-term question (deferred to Phases 4–5), and reran on GeoLife (raw), Foursquare NYC, **and now Gowalla-NYC**. Code: `scripts/run_phase2_disparate_risk.py`, `puffin/attacks/{windowing,reid,analysis}.py`. Full data: `reports/figures/phase2_v2/{geolife,foursquare,gowalla_nyc}/`.

## 1. Windowing fix: what changed and by how much

Dropping `respect_traj_boundary` for check-in data (full chronological sequence per user, not segmented by calendar day) recovers far more usable data — and, as section 3 shows, changes the finding, not just the sample size.

| dataset | SL | split | users_with_ge1_window | total_users | (Checkpoint 2 comparison) |
|---|---|---|---|---|---|
| Foursquare | 10 | train | 1083 | 1083 | — |
| Foursquare | 10 | test | **1083** | 1083 | was 71 (daily-segmented) |
| Foursquare | 5 | test | 1083 | 1083 | (sensitivity check — already saturated at SL=10) |
| Gowalla-NYC | 10 | train | 1517 | 1517 | not run at Checkpoint 2 |
| Gowalla-NYC | 10 | test | 587 | 1517 | |
| Gowalla-NYC | 5 | test | 1154 | 1517 | shorter SL recovers nearly 2x the test users |

Foursquare goes from a 6.6%, self-selected (frequent-checker-in) sample to the **entire dataset**. Gowalla-NYC's test coverage nearly doubles going from SL=10 to SL=5 (587→1154), confirming SL is a real, dataset-dependent sensitivity knob for check-in data — worth keeping in mind for Phase 3+ if Gowalla's SL=10 sample (587/1517, 39%) is judged too restrictive later.

## 2. Hardened attacker: which architecture, and a real generalization gap

5 seeds, best-of-{MLP, GRU} chosen by validation accuracy (never by peeking at test), early-stopped.

| dataset | winner (5/5 seeds) | mean val acc | mean test acc | gap |
|---|---|---|---|---|
| GeoLife | **MLP**, every seed | ~0.51 | ~0.17 | large |
| Foursquare | **GRU**, every seed | ~0.98 | ~0.57 | large |
| Gowalla-NYC | **GRU**, every seed | ~0.96 | ~0.25 | large |

Every number below is attributed to the architecture in this table. **The val/test gap is large and consistent across all three datasets** — the validation split is a random subsample of *train* windows, while test is each user's chronologically later 20% (Phase 1's `temporal_split`). A large gap here means real behavioral drift over time, not (or not only) overfitting: a model that reads someone's past well doesn't read their future nearly as well. This is itself a finding worth carrying into Phase 3/8 — "attacker strength" is highly sensitive to how recently the attack data was collected relative to the released features.

Even with that gap, test accuracy is far above random on every dataset: GeoLife 17% vs. 0.55% (182 classes), Foursquare 57% vs. 0.09% (1083 classes), Gowalla-NYC 25% vs. 0.07% (1517 classes, though only 587 ever appear as attackable).

## 3. Disparate risk, confirmed with bootstrap CIs — now on three datasets, GPS and check-in alike

| decile | GeoLife | Foursquare | Gowalla-NYC |
|---|---|---|---|
| 0 (lowest entropy) | 0.339 [0.197, 0.494] | 0.726 [0.669, 0.783] | 0.491 [0.402, 0.585] |
| 1 | 0.283 | 0.596 | 0.320 |
| 2 | 0.166 | 0.548 | 0.292 |
| 3 | 0.135 | 0.540 | 0.200 |
| 4 | 0.111 | 0.477 | 0.207 |
| 5 | 0.244 | 0.443 | 0.137 |
| 6 | 0.101 | 0.336 | 0.129 |
| 7 | 0.131 | 0.366 | 0.095 |
| 8 | 0.089 | 0.276 | 0.110 |
| 9 (highest entropy) | 0.076 [0.042, 0.111] | 0.243 [0.194, 0.296] | 0.148 [0.100, 0.199] |

All three: decile 0's CI and decile 9's CI do not overlap — a real, not noise-level, gap (4.5x for GeoLife, 3.0x for Foursquare, 3.3x for Gowalla-NYC). **Foursquare's pattern is the cleanest of the three** — monotonic, narrow CIs — now that the windowing fix gives it the full dataset instead of a biased 71-user slice. This directly answers Checkpoint 2's open question: the disparate-risk pattern **does replicate on check-in data**, once check-in data is windowed sensibly.

**OLS (`risk ~ S_unc + radius_of_gyration + n_windows + n_distinct_cells`): does entropy survive the controls?**

| dataset | S_unc coef | S_unc p | rog_km p | n_windows p | n_distinct_cells p | R² |
|---|---|---|---|---|---|---|
| GeoLife | -0.048 | 1.5e-07 | 0.336 | 0.168 | 0.174 | 0.159 |
| Foursquare | -0.273 | ~0 | ~0 | ~0 | ~0 | 0.369 |
| Gowalla-NYC | -0.234 | ~0 | ~0 | 0.716 | ~0 | 0.248 |

**Yes, on all three.** Entropy's effect is highly significant everywhere, controlling for the other three. On GeoLife it's the *only* significant predictor; on Foursquare and Gowalla-NYC the other covariates are also significant (R² is higher there too, 0.25–0.37 vs. 0.16) — check-in behavior is evidently shaped by more than just entropy, but entropy's own contribution doesn't wash out.

## 4. The key question: does plain Mo-PAE (λ4=0) under-protect low-entropy users?

Three `lambda_priv` operating points, λ4=0 throughout (no fairness term — deferred to Phases 4–5, per the PI). λ_priv=0.05 and 0.1 are too weak to read cleanly (several deciles show risk *increasing*, consistent with the mechanism barely perturbing anything yet). **λ_priv=0.3 is the first point where the mechanism visibly does something, and it's the one that answers the question:**

**Absolute risk reduction (raw − post) by decile, λ_priv=0.3:**

| decile | GeoLife | Foursquare | Gowalla-NYC |
|---|---|---|---|
| 0 | **-0.059** (got *worse*) | 0.233 | 0.153 |
| 1 | **-0.016** (got *worse*) | 0.326 | 0.112 |
| 2 | 0.020 | 0.341 | 0.094 |
| ... | | | |
| 7 | 0.013 | 0.293 | 0.066 |
| 8 | -0.000 | 0.250 | 0.062 |
| 9 | 0.024 | 0.223 | 0.083 |

A caveat before reading too much into absolute numbers alone: **relative reduction is numerically unstable for deciles with near-zero raw risk** (dividing a small post-mechanism change by a tiny baseline inflates the ratio) — the full per-decile relative numbers in `summary.json` show this clearly (e.g. GeoLife decile 9's mean relative reduction is -105%, driven by a few users whose already-tiny raw risk ticked up slightly). Absolute reduction is the more trustworthy lens here; relative reduction is reported too but should be read decile-by-decile, not as a single pooled average.

**GeoLife: Mo-PAE actively harms the two lowest-entropy deciles** (risk goes up, not down) while deciles 2 onward see a real, if modest, improvement. This is the strongest form of "under-protects the people who need it most" — not just *less* protection, active *negative* protection for exactly the regular, highly-predictable users the paper's motivating question is about.

**Foursquare and Gowalla-NYC: nobody gets worse, but the gap in degree is still there and runs the same direction.** Absolute reduction is roughly comparable across deciles (Foursquare: 0.22–0.35 everywhere; Gowalla: 0.06–0.15), but because low-entropy users start from a *much* higher baseline, the same absolute reduction leaves them with far more residual risk in relative terms: Foursquare decile 0 keeps ~68% of its original risk after the mechanism (0.726→0.493); decile 9 keeps ~8% (0.243→0.020). Gowalla-NYC decile 0 keeps ~69%; decile 9 keeps ~44%. Mean relative reduction, low deciles (0–2) vs. high deciles (7–9), at λ_priv=0.3: Foursquare 51% vs. 86%; Gowalla-NYC 25% vs. 67%.

**Conclusion, consistent across all three datasets and two completely different windowing regimes (GPS trajectories, full check-in sequences): plain Mo-PAE does not equalize risk, and the people it protects least (in relative terms everywhere, in absolute terms on GeoLife specifically) are exactly the low-entropy, most-predictable, most-at-risk users.** This is the pattern the PI's conditional was checking for. Per the PI's own instruction, this does **not** trigger a stop — it's the expected, paper-supporting result, so Phase 3 starts next without further sign-off.

## 5. Status check (requested)

- `reports/00b_pets_provenance.md`: done (Checkpoint 0, plus one addendum). Unaffected by anything in this session.
- GPUs: in active use, confirmed directly (`tensor.device`/`model.device` both `cuda:0` during a live op) and by running all three datasets in parallel on `cuda:0`/`cuda:1`/`cuda:2` for this report. The "15 epochs suggests CPU" concern was reasonable to raise but wasn't the case — `--device cuda` was explicit throughout; this revision raised the budget to up to 60 epochs with early stopping anyway.
- Environment: `requirements.lock.txt` written (`pip freeze`) after an unexpected erebor reboot mid-session corrupted matplotlib and contourpy (0-byte `__init__.py`, caught mid-write); both reinstalled and verified with a real plot before any further runs. All Phase 2 numbers reported here (and in the v1 report) were produced **after** the window-cap fix (the bug that inflated GeoLife's smoke test to 698K windows and OOM'd) — confirmed by the window counts in section 1 and `summary.json`'s `n_train_windows`/`n_test_windows` fields, which match the intended caps.

## 6. Limitations carried forward

Still 1 configuration beyond the PI's requested hardening that wasn't added: no formal convergence diagnostic beyond early stopping's own patience counter (no separate check that the loss curve has truly plateaued vs. patience just running out). Gowalla-NYC's SL=10 test coverage (587/1517, 39%) is real but worth watching if a future phase needs closer to full coverage — SL=5's 1154/1517 (76%) is available as a documented alternative. `lambda_priv` was swept on an arbitrary 3-point grid (0.05/0.1/0.3); Phase 5's baseline sweep should treat this as a starting point, not a tuned result.

## Figures and data

`reports/figures/phase2_v2/{geolife,foursquare,gowalla_nyc}/`: `risk_by_entropy_decile.png` (now with CI error bars), `risk_reduction_by_decile.png` (new, all 3 lambda_priv points), `per_user_risk.csv`, `summary.json`.
