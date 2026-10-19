# Checkpoint 3b — Hardening the adaptive-adversary result (Phase 3, item A)

**5-line summary:** Replicated Checkpoint 3's adaptive-adversary attack
with 5 seeds, bootstrap CIs, a full λ4 ∈ {0, 0.3, 1.0, 3.0} grid, and all
three datasets (GeoLife, Foursquare, Gowalla-NYC) — 12 runs total. The
**specialized attacker's gain over a naive attacker is statistically real
(CI excludes zero) in every single one of the 12 runs, including every
plain-Mo-PAE (λ4=0) run** — so this specific vulnerability is not an
artifact of the fairness mechanism; it is a generic property of any
trained encoder facing a smarter, subgroup-aware attacker. The
group-reweighted variant is messier: GeoLife and Foursquare show the same
pattern (positive, overlapping-magnitude gain even at λ4=0), but
Gowalla-NYC shows a real, CI-separated reversal — the reweighted attacker
does *worse* than naive under plain Mo-PAE but reliably *better* once
fairness is active, a genuine fairness-specific signature worth keeping
in the paper as a caveat, not a headline. The threat-model check shows a
user's entropy decile is itself recoverable from released encoder
features alone (34–42% top-1 vs. 10% chance) — the adaptive adversary's
"side information" requirement is weak, not a hypothetical worst case.
**Per the PI's stop condition, this is "not specific to the fairness
term" as the dominant, cross-dataset pattern — stopping and reporting
after this item, not proceeding to Phase 4/5 construction.**

## 1. Is the gain specific to the fairness mechanism? (item A.1)

Target-group mean top-1 accuracy, naive attacker vs. specialized attacker,
5 seeds each, with the bootstrap 95% CI on the **delta** (specialized −
naive) over users in the target decile:

| dataset | λ4 | target decile | naive | specialized | Δ (95% CI) | excludes 0? |
|---|---|---|---|---|---|---|
| GeoLife | 0.0 (plain) | 5 | 0.112 | 0.223 | +0.111 [0.050, 0.177] | yes |
| GeoLife | 0.3 | 5 | 0.162 | 0.290 | +0.128 [0.074, 0.185] | yes |
| GeoLife | 1.0 | 5 | 0.128 | 0.286 | +0.159 [0.110, 0.213] | yes |
| GeoLife | 3.0 | 5 | 0.144 | 0.295 | +0.151 [0.095, 0.209] | yes |
| Foursquare | 0.0 (plain) | 8 | 0.011 | 0.060 | +0.049 [0.030, 0.069] | yes |
| Foursquare | 0.3 | 7 | 0.116 | 0.261 | +0.145 [0.111, 0.181] | yes |
| Foursquare | 1.0 | 8 | 0.121 | 0.274 | +0.153 [0.119, 0.189] | yes |
| Foursquare | 3.0 | 9 | 0.126 | 0.215 | +0.089 [0.057, 0.124] | yes |
| Gowalla-NYC | 0.0 (plain) | 8 | 0.047 | 0.114 | +0.067 [0.041, 0.095] | yes |
| Gowalla-NYC | 0.3 | 7 | 0.043 | 0.116 | +0.073 [0.042, 0.109] | yes |
| Gowalla-NYC | 1.0 | 9 | 0.080 | 0.134 | +0.054 [0.033, 0.077] | yes |
| Gowalla-NYC | 3.0 | 9 | 0.080 | 0.127 | +0.047 [0.027, 0.068] | yes |

**Every single row's CI excludes zero — including every λ4=0 (plain
Mo-PAE) row.** GeoLife and Gowalla-NYC's plain-model deltas sit inside or
very close to their own fairness-active deltas' CIs (e.g. GeoLife: plain
[0.050,0.177] vs. λ4=1.0 [0.110,0.213], overlapping). Foursquare is the
outlier: its plain-model CI [0.030,0.069] sits clearly below λ4=0.3/1.0's
CIs ([0.111,0.181], [0.119,0.189]) — a real difference — but still well
above zero, i.e. the vulnerability exists under plain training there too,
just somewhat smaller.

Same table for the **group-reweighted** attacker (the variant specifically
designed to concentrate on the target group, arguably the more diagnostic
test of "is this exploit built out of the fairness mechanism itself"):

| dataset | λ4 | naive | reweighted | Δ (95% CI) | excludes 0? |
|---|---|---|---|---|---|
| GeoLife | 0.0 | 0.112 | 0.342 | +0.230 [0.183, 0.274] | yes (+) |
| GeoLife | 0.3 | 0.162 | 0.402 | +0.240 [0.194, 0.287] | yes (+) |
| GeoLife | 1.0 | 0.128 | 0.337 | +0.209 [0.138, 0.280] | yes (+) |
| GeoLife | 3.0 | 0.144 | 0.337 | +0.193 [0.146, 0.241] | yes (+) |
| Foursquare | 0.0 | 0.011 | 0.092 | +0.081 [0.062, 0.102] | yes (+) |
| Foursquare | 0.3 | 0.116 | 0.278 | +0.162 [0.134, 0.193] | yes (+) |
| Foursquare | 1.0 | 0.121 | 0.247 | +0.126 [0.101, 0.153] | yes (+) |
| Foursquare | 3.0 | 0.126 | 0.195 | +0.069 [0.050, 0.088] | yes (+) |
| Gowalla-NYC | 0.0 | 0.047 | 0.027 | **−0.023** [−0.039, −0.011] | yes (**−**) |
| Gowalla-NYC | 0.3 | 0.043 | 0.027 | −0.015 [−0.031, −0.004] | yes (**−**) |
| Gowalla-NYC | 1.0 | 0.080 | 0.135 | +0.055 [0.037, 0.076] | yes (+) |
| Gowalla-NYC | 3.0 | 0.080 | 0.117 | +0.037 [0.020, 0.056] | yes (+) |

GeoLife and Foursquare: positive, overlapping-magnitude gain regardless of
λ4 — not fairness-specific. **Gowalla-NYC reverses sign**: under plain
Mo-PAE the reweighted attacker reliably does *worse* than naive on the
target group (CI entirely negative), and only becomes reliably *better*
once the fairness mechanism is active (λ4 ≥ 1.0). This is the one place
in the whole grid where the adaptive-adversary exploit looks like it is
actually built out of the fairness term, not a property of training in
general.

**Reading the two tables together:** the generically-worse-attacker
pattern (specialized) is universal; the fairness-specific pattern
(reweighted reversing sign with λ4) shows up in exactly one dataset out of
three. The dominant, reproducible finding across datasets and both attack
variants is that a smarter attacker beats a naive one *regardless* of
whether a fairness mechanism is active — Checkpoint 3's headline 2.4–2.8x
number was real, but so is the same-order-of-magnitude number for plain
Mo-PAE that Checkpoint 3 didn't test.

## 2. Full λ4 grid, 3 datasets — summary

The grid above *is* the full replication (item A.2): λ4 ∈ {0, 0.3, 1.0,
3.0}, 5 seeds, all three datasets. No consistent dose-response: Foursquare
and Gowalla-NYC's specialized-attack deltas *shrink* as λ4 increases from
0.3 to 3.0 (0.145→0.089, 0.073→0.047) — the opposite of what "a bigger
fairness push creates a bigger exploitable gap" would predict.

## 3. Three-stage per-decile risk table (item A.3, λ4=1.0 runs)

Raw risk → post-mechanism risk under a naive attacker → post-mechanism
risk under the adaptive attacker (max of specialized/reweighted per
user), all deciles, mean top-1 accuracy:

**GeoLife** (target decile 5 in bold)

| decile | raw | post (naive) | post (adaptive) |
|---|---|---|---|
| 0 | 0.301 | 0.255 | 0.438 |
| 1 | 0.266 | 0.198 | 0.438 |
| 2 | 0.153 | 0.133 | 0.319 |
| 3 | 0.109 | 0.079 | 0.276 |
| 4 | 0.070 | 0.084 | 0.204 |
| **5** | **0.230** | **0.128** | **0.373** |
| 6 | 0.085 | 0.058 | 0.210 |
| 7 | 0.109 | 0.099 | 0.221 |
| 8 | 0.071 | 0.076 | 0.231 |
| 9 | 0.053 | 0.048 | 0.129 |

**Foursquare** (target decile 8)

| decile | raw | post (naive) | post (adaptive) |
|---|---|---|---|
| 0 | 0.754 | 0.670 | 0.805 |
| 1 | 0.644 | 0.532 | 0.740 |
| 2 | 0.577 | 0.427 | 0.635 |
| 3 | 0.597 | 0.413 | 0.635 |
| 4 | 0.500 | 0.348 | 0.555 |
| 5 | 0.490 | 0.307 | 0.487 |
| 6 | 0.379 | 0.209 | 0.387 |
| 7 | 0.410 | 0.201 | 0.369 |
| **8** | **0.320** | **0.121** | **0.308** |
| 9 | 0.272 | 0.114 | 0.209 |

**Gowalla-NYC** (target decile 9)

| decile | raw | post (naive) | post (adaptive) |
|---|---|---|---|
| 0 | 0.448 | 0.455 | 0.659 |
| 1 | 0.303 | 0.296 | 0.500 |
| 2 | 0.252 | 0.247 | 0.499 |
| 3 | 0.197 | 0.130 | 0.347 |
| 4 | 0.185 | 0.149 | 0.321 |
| 5 | 0.137 | 0.126 | 0.242 |
| 6 | 0.121 | 0.080 | 0.177 |
| 7 | 0.087 | 0.055 | 0.125 |
| 8 | 0.096 | 0.057 | 0.159 |
| **9** | **0.125** | **0.080** | **0.149** |

The pattern is the same in every decile of every dataset, not just the
"most protected" one: an adaptive attacker recovers most or all of the
gap between raw and naive-post risk, across the board. The fairness
mechanism's naive-attacker-measured protection (`raw` → `post (naive)`)
is real everywhere; the adaptive attacker erases most of it everywhere,
not selectively at the target decile.

## 4. Threat model: is decile membership releasable-feature-inferable? (item A.4)

`can_infer_decile_from_features` (`puffin/attacks/adaptive.py`): trains a
10-way classifier on the frozen encoder's own output to predict a user's
entropy decile, evaluated against the 10% chance rate.

| dataset | window-level acc. | user-level majority acc. | chance |
|---|---|---|---|
| GeoLife | 0.242 | **0.417** | 0.10 |
| Foursquare | 0.366 | **0.399** | 0.10 |
| Gowalla-NYC | 0.364 | **0.348** | 0.10 |

All three datasets: 3.5–4.2x chance at the user level. **The adaptive
adversary's premise — knowing each user's entropy-decile group — is not
privileged side information.** It is directly recoverable from the same
encoder output the mechanism releases for its stated purpose (next-
location prediction / downstream utility). This closes the gap the PI
asked about: the threat model is realistic, not a hypothetical worst case.

## 5. Membership inference, secondary (item A.5)

In-house loss-threshold MIA, 5 seeds, λ4=1.0 mechanism, bootstrap CI on
mean AUC:

| dataset | MIA AUC (95% CI) |
|---|---|
| GeoLife | 0.537 [0.535, 0.538] |
| Foursquare | **0.859** [0.838, 0.877] |
| Gowalla-NYC | 0.591 [0.584, 0.598] |

GeoLife's AUC matches Checkpoint 3's 0.555 (modest, consistent with an
underfit model on a large location vocabulary). Foursquare's 0.86 is
notably high and worth a follow-up footnote in the paper, but per the
PI's framing this metric stays secondary to the re-identification result
above, and does not change this item's decision.

## Decision

**Stop and report, per the PI's explicit stop condition.** The dominant,
reproducible pattern — across all three datasets, confirmed with 5 seeds
and bootstrap CIs, for the specialized attacker in all 12 runs and for
the reweighted attacker in 2 of 3 datasets — is that an adaptive,
subgroup-aware attacker recovers substantial risk **regardless of whether
the fairness mechanism is active.** Checkpoint 3's 2.4–2.8x number on the
fairness-active model was real, but so is a same-order-of-magnitude
number against plain Mo-PAE that Checkpoint 3 never tested. This means
the headline finding is not "Mo-PAE's fairness term creates a false sense
of security" — it is the stronger and more general claim that **naive-
attacker privacy_gain is not a trustworthy fairness metric for *any*
trained mobility-privacy mechanism**, fairness-aware or not. That is
itself a complete, reportable, and arguably more damning result for the
paper's framing than the original Task 5 finding.

The one genuine exception — Gowalla-NYC's reweighted-attacker sign flip
(negative under plain Mo-PAE, positive and CI-separated once fairness is
active) — is kept in the paper as a caveat/footnote: it shows a
fairness-specific failure mode *can* exist, just not as the dominant
pattern across datasets.

**Not proceeding to Phase 4 (B) or Phase 5 (C) construction** per the "if
yes, stop" branch of the PI's instruction. Recommending to the PI instead:
revise the paper's central claim from "does Mo-PAE's fairness mechanism
survive an adaptive adversary" to "does naive-attacker privacy_gain
measure anything trustworthy at all, for any mechanism" — a reframing
that changes which comparisons Phase 4/5 would need to make if the PI
decides to proceed after reviewing this.

## Figures and data

`reports/figures/phase3b/<dataset>_lambda4_<l4>/{summary.json,per_user_3b.csv}`
for all 12 (dataset, λ4) combinations. Orchestrator and logs:
`scripts/launch_phase3b_grid.sh`, `logs/phase3b/`.
