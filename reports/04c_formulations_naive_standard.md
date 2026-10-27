# Checkpoint 04c — Phase 4 formulations re-analyzed under the naive/hardened standard

> **SUPERSEDED (2026-10-27):** The gap this report identifies below —
> no plain-Mo-PAE utility, no closeable leveling-down check, ed_soft
> excluded from the Pareto analysis — is now closed. See
> `reports/04d_approved_batch.md` for the completed analysis: plain
> Mo-PAE, the no-privacy reference, ed_soft, and GeoLife's fixed
> floor_loss, all with >=3 encoder seeds and real utility, the rebuilt
> Pareto fronts, and the leveling-down check (which, where computable,
> consistently found leveling down rather than improvement for the
> worst-off decile). This report's own numbers below are kept for
> their original purpose (documenting the gap) and are not updated in
> place.

**5-line summary:** Re-analyzed Phase 4's existing runs (no retraining)
using only the naive/hardened external attacker numbers already on
disk, per the PI's new primary evaluation standard. Built Pareto fronts
(utility vs. decile-0 risk, utility vs. held-out-seed-selected max-min
risk) for 18 of 21 (dataset, formulation) pairs; **the leveling-down
check could not be run at all (0 comparisons) and the plain-Mo-PAE
reference curve could not be overlaid on the utility axis, because no
existing run ever recorded plain Mo-PAE's utility** — this is the one
real "gap in the fronts" this item was asked to identify, and it blocks
the PI's two most important asks for this checkpoint. `ed_soft` is
entirely excluded from the Pareto analysis for the same reason
(Checkpoint 3b's driver never logged utility either). Per-user minimum
filter applied throughout (Gowalla-NYC: 272/587 pass, same severe
exclusion Checkpoint 3d found). **Stopping and reporting after this
item, per the PI's instruction — not launching the training that would
close this gap without sign-off first.**

## What this reused (no retraining)

- `reports/figures/phase4/<dataset>_<formulation>[_lambda<l4>]/per_user_phase4.csv`
  for the 6 new formulations (lipschitz, floor, variance, group_dro,
  per_user_minimax, group_adversarial) at lambda4 ∈ {0.3, 1.0, 3.0} — the
  original 21-run grid plus the 33-run tuning-equalization grid, both
  already committed.
- `reports/figures/phase3b/<dataset>_lambda4_<l4>/per_user_3b.csv` for
  `ed_soft` (lambda4 ∈ {0, 0.3, 1.0, 3.0}) and for the plain-Mo-PAE
  (lambda4=0.0) reference.
- The naive/hardened attacker's existing per-seed `post_top1_seed{0-4}`
  columns, already saved in every one of those files — used for the
  held-out max-min decile selection without retraining anything.

## The blocking gap (item 3's "say which")

**Neither `run_phase3_adaptive_adversary.py`/`run_phase3b_adaptive_robustness.py`
(ed_soft's source) nor any run of `run_phase4_formulations.py` for
lambda4=0 ever existed for plain Mo-PAE with utility logged** —
Checkpoint 3b's driver computed naive/adaptive risk only, never
`next_location` top-1. Phase 4's own driver (`run_phase4_formulations.py`)
*does* log utility, but was never run at lambda4=0 (plain Mo-PAE is, by
construction, the same regardless of which `fairness_fn` is passed when
lambda4=0, so no formulation's lambda4=0 point was ever generated
either). Models aren't checkpointed to disk anywhere in this project, so
there is no way to recover plain Mo-PAE's utility from existing
artifacts — **it requires new training.**

This blocks:
1. **The leveling-down check entirely** (0 of the planned comparisons
   could be computed — every attempt required a plain-Mo-PAE point
   within ±0.02 utility of a formulation's point, and no plain-Mo-PAE
   utility value exists to compare against).
2. **Overlaying plain Mo-PAE as a reference curve on the utility axis**
   of the Pareto fronts below — plain Mo-PAE's risk values exist (decile
   0's risk, max-min risk, from Checkpoint 3d/3b data), but without a
   utility coordinate they can't be placed on the same plot.
3. `ed_soft`'s entire Pareto front (0 of 3 lambda points usable, all
   missing utility).

**What would fix it**: exactly 3 new training runs (one plain Mo-PAE per
dataset, `run_phase4_formulations.py --formulation <any> --lambda4 0.0`
— the fairness term is zero regardless of which formulation name is
passed, so any one of them works) plus re-running `ed_soft`'s existing
4 lambda points through the same utility-logging driver (3 more runs,
reusing `run_phase4_formulations.py --formulation ed_soft` at
lambda4 ∈ {0, 0.3, 1.0, 3.0}) — 6 runs total, roughly matching the
cost of one batch of the existing grid (~15-20 minutes). **Not launched
here, per "stop and report after step 2."**

## Pareto fronts (utility vs. decile-0 risk, utility vs. held-out max-min risk)

Decile 0 (lowest entropy) and the held-out-seed-selected worst-off decile
coincide in **every single one of the 54 points below** — the worst
decile under the naive/hardened standard is always the lowest-entropy
one, with no exceptions across any dataset, formulation, or lambda4.
Columns: lambda4, utility, decile-0 risk, worst decile (held-out-selected),
max-min risk (held-out-measured), n users (post-filter).

**GeoLife** (178/180 users after filter)

| formulation | λ4=0.3 | λ4=1.0 | λ4=3.0 |
|---|---|---|---|
| lipschitz | u=0.488 r=0.361 mm=0.323 | u=0.531 r=0.331 mm=0.328 | u=0.459 r=0.337 mm=0.323 |
| floor | u=0.473 r=0.376 mm=0.357 | u=0.473 r=0.376 mm=0.357 | u=0.473 r=0.376 mm=0.357 |
| variance | u=0.532 r=0.344 mm=0.350 | u=0.484 r=0.375 mm=0.352 | u=0.546 r=0.341 mm=0.350 |
| group_dro | u=0.526 r=0.364 mm=0.367 | u=0.559 r=0.347 mm=0.346 | u=0.536 r=0.308 mm=0.304 |
| per_user_minimax | u=0.626 r=0.384 mm=0.406 | u=0.505 r=0.351 mm=0.316 | u=0.412 r=0.333 mm=0.305 |
| group_adversarial | u=0.681 r=0.333 mm=0.286 | u=0.660 r=0.333 mm=0.288 | u=0.145 r=0.118 mm=0.139 |

`group_adversarial` at λ4=0.3 has the *highest* utility (0.681) of any
point in the whole grid, paired with one of the *lowest* max-min risks
(0.286) — a favorable **trade-off** point, not a confirmed dominance
claim (per the PI's definition — utility ≥ and risk < another point,
beyond both points' CIs — which hasn't been checked here and, per the
gap above, can't yet be checked against a plain-Mo-PAE reference
either). λ4=3.0 collapses utility to near-useless (0.145) for a modest
further risk reduction — not a sensible operating point regardless.

**Foursquare** (1083/1083 users after filter)

| formulation | λ4=0.3 | λ4=1.0 | λ4=3.0 |
|---|---|---|---|
| lipschitz | u=0.075 r=0.401 mm=0.406 | u=0.106 r=0.400 mm=0.394 | u=0.092 r=0.481 mm=0.477 |
| floor | u=0.102 r=0.407 mm=0.414 | u=0.108 r=0.440 mm=0.438 | u=0.101 r=0.359 mm=0.362 |
| variance | u=0.100 r=0.455 mm=0.453 | u=0.109 r=0.413 mm=0.403 | u=0.100 r=0.422 mm=0.414 |
| group_dro | u=0.087 r=0.312 mm=0.305 | u=0.086 r=0.422 mm=0.424 | u=0.079 r=0.306 mm=0.303 |
| per_user_minimax | u=0.081 r=0.274 mm=0.278 | u=0.055 r=0.113 mm=0.118 | u=0.050 r=0.043 mm=0.056 |
| group_adversarial | u=0.088 r=0.521 mm=0.531 | u=0.043 r=0.354 mm=0.353 | u=0.036 r=0.033 mm=0.021 |

**Corrected per the PI (dominance requires utility ≥ and risk < beyond
both points' CIs, which this comparison does not check): re-labeled as
a trade-off, not a dominance claim.** At comparable utility (~0.08-0.11),
`group_dro` (r=0.306-0.312 at λ4∈{0.3,3.0}) and `per_user_minimax`
(r=0.274 at λ4=0.3) sit at lower point-estimate risk than
`floor`/`variance`/`group_adversarial`'s points in the same utility
range (r=0.4-0.52) — a real, cross-formulation difference in the point
estimates, visible without needing the plain-Mo-PAE reference, but not
a verified dominance relationship without CIs on both axes.

**Gowalla-NYC** (272/587 users after filter — Checkpoint 3d's severe
exclusion applies here too)

| formulation | λ4=0.3 | λ4=1.0 | λ4=3.0 |
|---|---|---|---|
| lipschitz | u=0.268 r=0.465 mm=0.459 | u=0.288 r=0.539 mm=0.497 | u=0.290 r=0.537 mm=0.531 |
| floor | u=0.241 r=0.499 mm=0.479 | u=0.236 r=0.511 mm=0.517 | u=0.232 r=0.488 mm=0.483 |
| variance | u=0.305 r=0.504 mm=0.481 | u=0.259 r=0.544 mm=0.535 | u=0.266 r=0.499 mm=0.506 |
| group_dro | u=0.295 r=0.514 mm=0.499 | u=0.225 r=0.508 mm=0.500 | u=0.248 r=0.533 mm=0.506 |
| per_user_minimax | u=0.214 r=0.499 mm=0.505 | u=0.156 r=0.466 mm=0.428 | u=0.091 r=0.408 mm=0.393 |
| group_adversarial | u=0.259 r=0.556 mm=0.546 | u=0.050 r=0.329 mm=0.318 | u=0.042 r=0.263 mm=0.263 |

No formulation clearly dominates here at comparable utility — risk is
uniformly high (0.41-0.56) across the whole grid, consistent with
Gowalla-NYC's generally noisier, smaller-sample behavior throughout this
project. `group_adversarial` only reduces risk by collapsing utility,
same pattern Checkpoint 4 found before the 3d correction.

## Leveling-down check: not computable (0 comparisons)

Every attempt to match a formulation's operating point to plain Mo-PAE
within ±0.02 utility failed because plain Mo-PAE has no recorded utility
value to match against (see the gap above). `reports/figures/phase4c/leveling_down.json`
is an empty list. This is a direct, unavoidable consequence of the same
gap, not a separate issue.

## Encoder-training seeds (per the PI's request to state this explicitly)

**Every point in the 54-point grid above (and the whole Phase 4 grid it
was built from) used exactly ONE encoder-training seed (seed=0) per
(dataset, formulation, λ4).** The 5-seed treatment throughout this
project's "attacker seeds" columns is multi-seed only on the ATTACKER
side (naive/adaptive attacker re-trained 5 times against the SAME
single frozen encoder) — the mechanism itself (the encoder, decoder,
id_recognizer, next_predictor) was trained once per operating point.
This means every number above carries encoder-initialization variance
that has never been measured. The PI's approved batch (plain Mo-PAE,
ed_soft, and the no-privacy reference) uses ≥3 encoder seeds each,
reporting means and CIs over encoder seeds as well as attacker seeds —
see the rebuilt fronts below. The other 54 points' 1-seed limitation is
noted here, not retroactively fixed (that would be a much larger rerun
than this round's scope).

## Per-user minimum filter

| dataset | before | after | excluded |
|---|---|---|---|
| GeoLife | 180 | 178 | 2 |
| Foursquare | 1083 | 1083 | 0 |
| Gowalla-NYC | 587 | 272 | 315 (53.7%) |

Consistent with Checkpoint 3d's finding on the same datasets under the
(now-retired) adaptive-attacker pipeline — Gowalla-NYC's coverage
problem is a property of the dataset/windowing, not the attacker.

## What step 3 needs (not run here)

Exactly the 6 runs named above (3 plain-Mo-PAE-with-utility, 3 ed_soft-
with-utility) would close every gap in this report. Recommending these
specifically as the "new training where step 2 shows gaps" the PI asked
for — awaiting sign-off before launching, per "stop and report after
step 2."

## Figures and data

`reports/figures/phase4c/{pareto_fronts,leveling_down,gaps}.json`.
Analysis script: `scripts/analyze_04c_naive_standard.py` (pure analysis,
no training).
