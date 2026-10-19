# Checkpoint 3 — Attack suite and metrics (Phase 3)

**5-line summary:** All five Phase 3 tasks are done. Of the two third-party
repos the plan named, one ran (MainTUL, after fixing two trivial upstream
bugs) and one didn't (POIPrivacy, for structural reasons — a ~29GB memory
wall and missing schema fields, not a dependency problem), so per the PI's
time-box MIA uses an in-house loss-threshold attack instead; both are
real, tested, and give non-trivial signal. The headline new result: a
fairness-aware adaptive adversary who knows which entropy group a
mechanism protects most recovers **2.4–2.8x more risk** for that exact
group than a naive attacker reports (18% → 31–37% top-1), on real GeoLife
data against an actually fairness-active Mo-PAE. This doesn't change the
project's framing — if anything it strengthens the paper's core claim that
measured "fairness" can be an artifact of a weak attacker — so per the
"proceed unless" rule, **continuing straight to Phase 5 baselines.**

## Task 1 — Threat model + metrics module

`puffin/attacks/README.md`: per-mechanism threat table (Mo-PAE/PUFFIN
releases encoder features → re-id + MIA; TrajGAN/GI-DP release trajectories
→ TUL + re-id/MIA). `puffin/metrics/{privacy,fairness,leveling_down}.py`:
`privacy_gain`, `protection_at_k`, `similar_pairs_within_epsilon`,
`lipschitz_violation`, `max_min_gain`, `gini`, `gain_percentiles`,
`leveling_down_check`; `puffin/attacks/analysis.py` gained
`paired_bootstrap_test`. All covered by `tests/test_metrics.py`.

## Task 2 — Finalized external re-id attacker

`train_raw_attacker_strongest_per_user` (`puffin/attacks/reid.py`): MLP +
GRU + gradient-boosted trees (sklearn `HistGradientBoostingClassifier`),
per-user element-wise max across all three — not one global per-seed
winner, per the plan's literal "report the strongest attacker per user."
Every per-user row records which architecture won. Checkpoints 2b/2c's
2-architecture, per-seed-winner function is left unchanged so those
already-reported numbers stay reproducible. Verified on real GeoLife data
(`scripts/smoke_test_phase3_attacker.py`).

## Task 3 — Membership inference

**POIPrivacy did not run on our data**, for reasons found by actually
reading its code, not guessing: its LocMIA/TrajMIA attacks are hardwired
to GETNext, a specific POI-recommender architecture, whose dense
POI-adjacency graph would need ~29GB for GeoLife's 85,074 cells (confirmed
by running its own demo and measuring the output), and whose POI-category
node features don't exist anywhere in this project's unified schema. Both
are independently exactly the plan's stated stop condition. Built instead:
`puffin/attacks/mia_inhouse.py`, a classic Yeom/Shokri loss-threshold MIA
attacking Mo-PAE-fixed's own next-location model, with a calibration split
(so the threshold that scores a window never used that same window to set
itself) and per-user output. Real GeoLife validation: overall AUC 0.555,
accuracy 0.598 (181 users) — modest and non-suspicious, exactly what an
underfit model on an 85K-location vocabulary should show. Tested:
`tests/test_mia_inhouse.py` (5 checks).

## Task 4 — Trajectory-user linking

**MainTUL did run**, unlike POIPrivacy — its own demo crashed on two
trivial bugs (`np.Inf`, removed in numpy≥2.0; a logging format-string
mismatch that crashes the authors' own default config on the first logged
batch), both one-line fixes, after which it trained and tested
successfully end to end. Ported its mutual-distillation LSTM-student /
Transformer-teacher architecture into `puffin/attacks/tul_maintul.py`
(a from-scratch port, not an import of the uncommitted clone), dropping
only the POI-category view (same missing-schema reason as task 3). Per the
threat model, TUL only applies to mechanisms that release trajectories
(TrajGAN, GI-DP) — neither exists yet (that's Phase 5), so this validated
the attack module itself against GeoLife's own trajectories as a
stand-in. Real validation (163 non-disjoint users): student top-1 0.557,
top-5 0.795 (chance ≈0.006) — large and real, with the SAME disparate
pattern as every other checkpoint in this project: median per-user top-1
is only 0.071, with pooled accuracy dominated by a handful of
very-prolific users. Tested: `tests/test_tul_maintul.py` (4 checks).

## Task 5 — Fairness-aware adaptive adversary

The question: does a mechanism's apparent fairness (measured by a naive,
group-agnostic attacker) survive an adversary who knows the fairness
objective and each user's entropy group? Trained an actually
fairness-ACTIVE Mo-PAE (`ed_soft`, `lambda4=1.0`, `lambda_priv=0.3` — unlike
every plain-Mo-PAE run in Checkpoints 2/2b/2c, which keep `lambda4=0`).
Found its most-protected decile via the ordinary attacker
(`puffin.metrics.privacy.privacy_gain`), then attacked the same frozen
encoder with two adaptive variants (`puffin/attacks/adaptive.py`):

| attacker | target decile (5) mean top-1 | all-users mean Δ vs. naive |
|---|---|---|
| naive, group-agnostic | **0.129** | — |
| (a) per-group specialized | 0.306 (**2.4x**) | +0.169 (helps broadly — expected; smaller specialized models often do) |
| (b) group-reweighted | 0.365 (**2.8x**) | +0.003 (~0 — correctly concentrated on the target group only, exactly as designed) |

Checked for the near-zero-denominator instability flagged in Checkpoint
2c's relative-metric caveat before trusting this: one other decile's raw
`privacy_gain` mean (-4.69) is entirely a single user's artifact
(raw_top1=0.001, post_top1=0.077 → gain=-76); decile 5 itself has no such
outlier (individual gains range -0.94 to +1.0, mean-of-gains 0.475 vs.
ratio-of-means 0.438 — consistent, not inflated), and the headline
0.129/0.306/0.365 numbers are plain top-1 accuracy means, not ratios, so
immune to that specific failure mode regardless.

**This is a real, substantial, and — per the "proceed unless it changes
the framing" rule — non-stopping result.** It doesn't contradict anything
established so far; it sharpens it: Mo-PAE's fairness mechanism can look
like it's protecting a group well when evaluated the way Checkpoints
2/2b/2c did, while a smarter, fairness-aware adversary shows that
protection is partly an artifact of not having been tested against the
right attacker. That is exactly "the old paper's own limitation" the plan
named this task to answer.

## Decision

Proceeding straight to Phase 5 baselines, per the PI's standing "proceed
unless something changes the framing" rule — nothing here changes it.

## Figures and data

`reports/figures/phase3_adaptive/geolife/{summary.json,per_user_adaptive.csv}`.
