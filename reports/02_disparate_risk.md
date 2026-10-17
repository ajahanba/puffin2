# Checkpoint 2 — Disparate-risk motivation

**Status: ⛔ stopping here per the plan, awaiting PI review before Phase 3.**

**Question:** is re-identification risk unequal across users in a way that matters, and does Mo-PAE (fixed) protect the people who need it most — or does it just equalize by weakening protection for everyone?

Run on Foursquare NYC and raw GeoLife 1.3, per the PI's instruction. Code: `scripts/run_phase2_disparate_risk.py`, `puffin/features/mobility.py`, `puffin/attacks/{windowing,reid}.py`. Raw data/figures: `reports/figures/phase2/{geolife,foursquare}/`.

## Method

1. **Per-user mobility features** (Song et al. 2010): random entropy `S_rand = log2(n_distinct_cells)`, uncorrelated entropy `S_unc` (Shannon entropy of visitation frequency), real/actual entropy `S_real` (Lempel-Ziv entropy-rate estimator on each user's run-length-collapsed visit sequence, capped at 3,000 visits — GeoLife's longest user trace is 2.1M raw GPS points, far too long for this estimator uncapped). Plus radius of gyration (haversine distance from each record to the user's center of mass) and regularity (share of all visits on the user's own top-2 cells).
2. **Raw re-id attacker**: a fresh encoder+classifier (embedding + 2-layer LSTM + MLP head, the same architecture Phase 0/1 already established) trained directly on unprotected SL=10 windows to predict user identity. This stands in for Phase 3's full attack suite, which isn't built yet — it's the strongest single attacker reasonably available at this stage, not the final word on "worst-case" risk.
3. **OLS**: `risk ~ S_unc + radius_of_gyration + n_records` (plain least squares, no new dependency).
4. **Mo-PAE (fixed) at 4 operating points** (`lambda4` ∈ {0.0, 0.3, 0.7, 1.5}, the fairness-term weight on `ed_soft`, the Phase 4 `L_pair_threshold` baseline from the earlier fairness-loss port) — each followed by a **fresh** attacker trained on that run's **frozen** encoder output (never the in-model adversary), per Phase 3.2.1's methodology.

**Scale-down from a production run, flagged explicitly:** 1 seed (not the ≥5 the ground rules call for), 15 epochs each for the attacker and for Mo-PAE, windows capped at 200/user. This is a motivation-level check, not a result to cite.

## GeoLife (raw 1.3, 180 of 182 users usable)

**Risk is clearly unequal, and the gradient runs the direction the plan hoped for.** Mean raw top-1 risk is 14.2% (vs. 0.55% random-guess baseline for 182 classes) — already far above chance. By entropy decile (0 = most predictable/regular, 9 = most random):

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean risk | **0.297** | 0.232 | 0.148 | 0.123 | 0.077 | 0.207 | 0.079 | 0.105 | 0.081 | **0.069** |

Decile 0 is **4.3x** decile 9. Checked this isn't a one-user fluke (`per_user_risk.csv`): several distinct low-entropy users hit raw_top1 = 1.0 or close to it, not a single outlier dragging the mean. The OLS fit is real but weak (R²=0.127, `S_unc` coefficient -0.035) — the decile pattern is visible but noisy, not a clean straight line (decile 5's blip above deciles 2-4 and 6-8 is the main irregularity).

**Who gains/loses under Mo-PAE (fixed):**

| lambda4 | mean Δrisk | helped | harmed | unchanged | decile-0 Δ | decile-9 Δ |
|---|---|---|---|---|---|---|
| 0.0 (no fairness term) | -0.034 | 57% | 27% | 16% | **+0.005** (unhelped) | -0.039 |
| 0.3 | -0.010 | 50% | 34% | 16% | **-0.050** (helped) | -0.032 |
| 0.7 | -0.017 | 49% | 37% | 14% | -0.005 | -0.019 |
| 1.5 | -0.026 | 54% | 24% | 22% | -0.022 | -0.020 |

With the fairness term **off** (`lambda4=0.0`), the worst-off decile (0) is essentially untouched (+0.005, within noise) while several *already-safer* deciles improve more — privacy gain is distributed roughly without regard to who needed it. Turning the fairness term **on** (`lambda4=0.3`) flips decile 0 from "unhelped" to the *most*-helped decile. That's the encouraging direction. But the effect does not scale cleanly with `lambda4`'s magnitude (0.7 helps decile 0 almost nothing; 1.5 is in between) — at 15 epochs and 1 seed, this reads as "the fairness term can help the worst-off, not that it reliably does at any given weight." Nothing here is yet evidence against leveling-down either way; a real test needs the ≥5-seed, matched-utility comparison the plan's Phase 3 metrics call for (`ΔP(u)` at matched Acc@5).

## Foursquare NYC (71 of 1,083 users usable — a real limitation, not a footnote)

Only 71 users end up in the risk analysis at all, because Foursquare's trajectories are segmented by calendar day (no native trajectory id, see `puffin/data/datasets/_checkin_utils.py`) and SL=10 requires 10+ check-ins in a single day — something only ~21% of users ever do even once (`windows_at_sl10.users_with_ge1_window=231/1083` in Phase 1's stats.json), and fewer still end up with a usable window in *both* the train and test split. The 71 users who do clear that bar are, by construction, an unusually frequent-checking-in subpopulation — not representative of Foursquare users generally.

Within that constrained sample: mean raw risk 1.4% (vs. ~0.09% true random baseline over 1,083 classes — still a real lift, roughly 15x). The decile table is almost entirely zero except decile 0 (0.125) — with only ~7 users per decile and most deciles having zero successes, this is too sparse to read as a trend one way or the other. The Mo-PAE operating-point deltas are similarly dominated by small-sample noise (`frac_unchanged` is 94-97%, meaning almost nobody's risk moved at all, consistent with most users here having only 1-2 test windows total).

**Foursquare's finding, honestly stated: the data does not support a disparate-risk conclusion at SL=10 with daily trajectory segmentation.** This is a data/methodology constraint, not evidence that Foursquare's risk is actually flat. A shorter SL (e.g. 5) or a non-daily trajectory definition would very likely recover a usable sample, but changing SL for only one dataset wasn't done here without checking with the PI first, per the project's "no silent fixes" rule.

## Figures

`reports/figures/phase2/geolife/risk_by_entropy_decile.png`, `risk_vs_regularity.png`; same two files under `.../foursquare/` (the Foursquare versions should be read with the sample-size caveat above in mind). Per-user data: `per_user_risk.csv` in each directory.

## Recommendation for this checkpoint

**GeoLife supports the plan's motivating claim** — regular/predictable users are substantially more re-identifiable from raw data, and there's a real (if not yet robust) signal that a fairness-weighted mechanism can specifically help that worst-off group rather than leveling everyone down. This is enough to justify continuing to Phase 3/4 on GeoLife.

**Foursquare does not yet support or refute it** — the SL=10 + daily-segmentation combination leaves too small and too biased a sample. Before using Foursquare for this paper's central claim, the PI should decide: shorten SL for check-in-style datasets specifically, redefine trajectories (e.g., sliding time windows instead of calendar days), or treat Foursquare as a secondary/robustness dataset rather than a primary one for the disparate-risk story.

Per the plan's own checkpoint rule, stopping here rather than proceeding into Phase 3.
