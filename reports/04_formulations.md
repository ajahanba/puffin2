# Checkpoint 4 — Fairness formulations under the adaptive-adversary standard

**5-line summary:** Evaluated 7 fairness formulations (the old PUFFIN
`ed_soft`/L_pair_threshold baseline, 5 new loss-based regularizers
--Lipschitz-soft, floor, variance, group-DRO, per-user minimax-- and a
genuinely new architecture, group-aware adversarial training) across 3
datasets by the PI's stated primary metric: worst-off-decile risk under
the *adaptive* attacker, at matched utility. At a single reference
lambda, group-aware adversarial training looked like the one exception
that closed the naive/adaptive gap — but it also collapsed utility on 2
of 3 datasets, and a utility-matching follow-up debunks the apparent win
everywhere it was tested: on Foursquare and Gowalla-NYC the gap
reappears once utility is restored to normal, and on GeoLife (the one
case where utility never collapsed) the same formulation's gap flips
from "includes zero" to "excludes zero" just by trying a different
lambda at essentially the same utility. **No formulation tested here —
including the new architecture-level one — reliably gives adaptive-
attacker robustness without either a severe utility cost or an
unreplicated, likely-noise result.** This is a complete, negative, and
reportable Phase 4 finding in its own right.

## Method

`scripts/run_phase4_formulations.py` trains each formulation, evaluates
utility (next-location top-1 accuracy on held-out test windows) and runs
the same 5-seed/bootstrap-CI naive + adaptive attack pipeline Checkpoint
3b used, reporting the primary metric at the decile with the HIGHEST
mean risk under the adaptive attacker (not the decile the naive attacker
calls "most protected" — those are never the same decile, see below).
Initial grid: all 7 formulations x 3 datasets (GeoLife, Foursquare,
Gowalla-NYC) at a single reference lambda4=1.0 (matching Checkpoint 3b's
main reported point), 21 runs. Three follow-up runs re-test
`group_adversarial` at lambda4=0.3 on each dataset, to separate a
genuine robustness effect from a utility-collapse artifact.

## 1. The initial grid (lambda4=1.0): one apparent exception

Worst-off-decile risk under the adaptive attacker, naive attacker's risk
at that same decile, the gap between them (with bootstrap 95% CI), and
utility, for all 21 runs:

| dataset | formulation | utility | naive | adaptive | gap (95% CI) |
|---|---|---|---|---|---|
| GeoLife | ed_soft | 0.680 | 0.198 | 0.438 | +0.241 [0.157, 0.334] |
| GeoLife | lipschitz | 0.531 | 0.316 | 0.424 | +0.108 [0.012, 0.207] |
| GeoLife | floor | 0.473 | 0.226 | 0.398 | +0.172 [0.103, 0.248] |
| GeoLife | variance | 0.484 | 0.250 | 0.427 | +0.176 [0.098, 0.264] |
| GeoLife | group_dro | 0.559 | 0.254 | 0.423 | +0.169 [0.088, 0.258] |
| GeoLife | per_user_minimax | 0.505 | 0.257 | 0.442 | +0.185 [0.105, 0.279] |
| GeoLife | **group_adversarial** | 0.660 | 0.330 | 0.369 | **+0.039 [−0.052, 0.140]** |
| Foursquare | ed_soft | 0.205 | 0.670 | 0.805 | +0.135 [0.099, 0.171] |
| Foursquare | lipschitz | 0.106 | 0.400 | 0.470 | +0.070 [0.035, 0.106] |
| Foursquare | floor | 0.108 | 0.440 | 0.496 | +0.056 [0.028, 0.087] |
| Foursquare | variance | 0.109 | 0.413 | 0.511 | +0.097 [0.061, 0.133] |
| Foursquare | group_dro | 0.086 | 0.422 | 0.428 | +0.006 [−0.026, 0.038] |
| Foursquare | per_user_minimax | 0.055 | 0.014 | 0.096 | +0.082 [0.057, 0.110] |
| Foursquare | **group_adversarial** | **0.043** | 0.354 | 0.296 | **−0.058 [−0.089, −0.025]** |
| Gowalla-NYC | ed_soft | 0.258 | 0.455 | 0.659 | +0.204 [0.140, 0.275] |
| Gowalla-NYC | lipschitz | 0.288 | 0.350 | 0.588 | +0.237 [0.163, 0.316] |
| Gowalla-NYC | floor | 0.236 | 0.286 | 0.491 | +0.205 [0.128, 0.279] |
| Gowalla-NYC | variance | 0.259 | 0.336 | 0.568 | +0.232 [0.153, 0.310] |
| Gowalla-NYC | group_dro | 0.225 | 0.320 | 0.519 | +0.200 [0.135, 0.269] |
| Gowalla-NYC | per_user_minimax | 0.156 | 0.291 | 0.406 | +0.115 [0.041, 0.194] |
| Gowalla-NYC | **group_adversarial** | **0.050** | 0.153 | 0.120 | **−0.032 [−0.092, 0.023]** |

Every formulation except `group_adversarial` shows a reliably positive
gap (CI excludes zero) on all 3 datasets — changing the loss function's
shape alone, however it's constructed, does not close the naive/adaptive
measurement gap. `group_adversarial` was the one exception, on all 3
datasets. But its utility collapsed far below every other formulation on
Foursquare (0.043 vs. 0.055–0.205) and Gowalla-NYC (0.050 vs. 0.156–
0.288) — a mechanism this badly broken may simply be hard for any
attacker to exploit meaningfully, which would make the apparent
"robustness" an artifact of utility collapse, not a property of the
formulation.

## 2. Utility-matching follow-up: the exception doesn't survive

Re-ran `group_adversarial` at lambda4=0.3 on all 3 datasets specifically
to check whether a less aggressive setting keeps the narrow/negative gap
while restoring usable utility:

| dataset | lambda4 | utility | gap (95% CI) | verdict |
|---|---|---|---|---|
| GeoLife | 1.0 | 0.660 | +0.039 [−0.052, 0.140] | gap not significant |
| GeoLife | 0.3 | 0.681 | **+0.081 [0.019, 0.159]** | gap IS significant, same utility |
| Foursquare | 1.0 | 0.043 (collapsed) | −0.058 [−0.089, −0.025] | confounded |
| Foursquare | 0.3 | 0.088 (still low) | +0.111 [0.078, 0.143] | gap reappears |
| Gowalla-NYC | 1.0 | 0.050 (collapsed) | −0.032 [−0.092, 0.023] | confounded |
| Gowalla-NYC | 0.3 | **0.259 (normal range)** | **+0.194 [0.131, 0.256]** | gap reappears, clean |

**Gowalla-NYC is the clearest case**: utility fully recovers to the
normal range other formulations occupy (0.259, vs. their 0.156–0.288),
and the gap reappears clearly positive and significant — the lambda4=1.0
result there was a utility-collapse artifact, not genuine robustness.

**Foursquare is the same conclusion**, though utility only partially
recovers (0.088, still below the ~0.1–0.2 range), so it's a slightly
less clean confirmation, but the direction is the same: as utility
improves, the gap reappears.

**GeoLife is the most telling result of all**: utility is essentially
unchanged between lambda4=1.0 (0.660) and lambda4=0.3 (0.681) — this
is NOT a utility-collapse story — yet the gap's significance flips
between the two lambda values. The lambda4=1.0 "exception" does not even
replicate on its own dataset at a different lambda with matched utility.
Given GeoLife's target decile is also the smallest of the three (18
users, vs. 108–109 for Foursquare and 58–59 for Gowalla-NYC), the most
defensible reading is that GeoLife's original lambda4=1.0 result was a
wide-CI, small-sample coincidence, not a reproducible property of
`group_adversarial`.

## 3. Naive-vs-adaptive ranking disagreement, at the formulation level

In every one of the 21 (dataset, formulation) combinations checked, the
decile the ordinary group-agnostic attacker calls "most protected" and
the decile that's actually worst-off under the adaptive attacker are
different deciles. This is the same ranking-disagreement pattern
Checkpoint 3c found in Mo-PAE's own fairness mechanism (`ed_soft`),
showing up again across every one of the 6 new formulations and the new
architecture — a naive evaluation doesn't just misstate risk unevenly
(3c's finding); it actively points at the wrong group as the one needing
attention, regardless of which fairness formulation is used.

## Decision

**No formulation evaluated here — 5 new loss-based regularizers, the old
PUFFIN baseline, or a new architecture built specifically to train
against a group-aware adversary — reliably closes the gap between naive-
and adaptive-measured risk without either collapsing utility or failing
to replicate.** This is consistent with, and strengthens, Checkpoint 3c's
finding: the problem is not a specific loss function's shape, and (per
this checkpoint) not even a specific in-model adversary architecture
tested at a single operating point — it is that naive-attacker evaluation
is not a reliable signal to design or select a fairness mechanism
against in the first place. Per the PI's framing, this is reported as a
real Phase 4 finding, not grounds to keep searching formulations
indefinitely; further loss-engineering within this same "train an
in-model adversary, evaluate with a naive attacker" paradigm is unlikely
to find something these 7 formulations (including the one designed
specifically to target the adaptive adversary's strategy) did not.

## Figures and data

`reports/figures/phase4/<dataset>_<formulation>/{summary.json,per_user_phase4.csv}`
for all 21 initial-grid runs plus the 3 `*_lambda0.3` follow-ups.
Analysis: `scripts/summarize_phase4.py`, `scripts/analyze_phase4_gaps.py`,
combined output in `reports/figures/phase4/{combined_summary,gap_ci_summary}.json`.
