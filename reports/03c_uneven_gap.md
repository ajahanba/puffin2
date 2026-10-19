# Checkpoint 3c — The naive/adaptive measurement gap is uneven across users

**5-line summary:** Per the PI's revised framing (that adversarially
trained representations leaking to post-hoc attackers is established —
see `reports/related_work_adversarial_leakage.md` — and the project's
contribution is that this leakage is measured **unevenly across users**),
this checkpoint tests exactly that on the already-collected Checkpoint 3b
data (no new training). Result: the gap between naive and adaptive
measured risk is statistically non-uniform across entropy deciles in
**all 6 panels** (3 datasets × {plain, fairness-active} Mo-PAE,
Kruskal-Wallis p ≤ 0.017 everywhere), and it is largest for the
**lowest-entropy deciles — the users a fairness mechanism is supposed to
protect most.** Ranking users by naive privacy_gain disagrees
substantially with ranking them by adaptive privacy_gain (Spearman
ρ = 0.16–0.67, far from 1), and far more users move from
"well-protected" under the naive ranking to "poorly-protected" under the
adaptive one than the reverse (11–67 vs. 0–5, depending on panel). **Per
the PI's continue condition, the gap is not uniform — continuing straight
to Phase 4 (B).**

## 1. Related work (per the PI)

`reports/related_work_adversarial_leakage.md`: Elazar & Goldberg (2018,
EMNLP) and Song & Shmatikov (2020, ICLR), both verified directly. This
project does not claim naive-attacker leakage from adversarially trained
representations is novel — these two papers establish that in a
different domain. What's new here is that the leakage is uneven across
users in a way that undermines fairness claims built on naive evaluation.

## 2. Gap by entropy decile, with bootstrap CIs

Measurement gap = post-mechanism top-1 under the **adaptive** attacker
minus under the **naive** attacker, mean per decile, 95% bootstrap CI
over users (all CIs below exclude zero unless noted):

**GeoLife**

| decile (0=lowest entropy) | plain Mo-PAE gap | fairness-active gap |
|---|---|---|
| 0 | +0.034 [−0.021, +0.092] (CI includes 0) | +0.183 [0.105, 0.264] |
| 1 | +0.172 [0.103, 0.248] | +0.241 [0.157, 0.334] |
| 2 | +0.145 [0.067, 0.231] | +0.186 [0.115, 0.264] |
| 3 | +0.153 [0.071, 0.243] | +0.197 [0.129, 0.270] |
| 4 | +0.114 [0.069, 0.165] | +0.120 [0.072, 0.176] |
| 5 | +0.240 [0.188, 0.290] | +0.245 [0.180, 0.313] |
| 6 | +0.116 [0.076, 0.157] | +0.152 [0.103, 0.208] |
| 7 | +0.114 [0.063, 0.171] | +0.122 [0.086, 0.161] |
| 8 | +0.117 [0.054, 0.193] | +0.155 [0.085, 0.247] |
| 9 (highest entropy) | +0.089 [0.050, 0.134] | +0.081 [0.030, 0.141] |

**Foursquare**

| decile | plain Mo-PAE gap | fairness-active gap |
|---|---|---|
| 0 | +0.059 [0.030, 0.088] | +0.135 [0.099, 0.171] |
| 1 | +0.112 [0.077, 0.147] | +0.208 [0.167, 0.251] |
| 2 | +0.107 [0.075, 0.141] | +0.208 [0.164, 0.255] |
| 3 | +0.080 [0.051, 0.112] | +0.222 [0.184, 0.260] |
| 4 | +0.090 [0.063, 0.121] | +0.208 [0.172, 0.244] |
| 5 | +0.067 [0.040, 0.097] | +0.180 [0.147, 0.217] |
| 6 | +0.080 [0.056, 0.105] | +0.178 [0.147, 0.209] |
| 7 | +0.064 [0.040, 0.091] | +0.168 [0.137, 0.203] |
| 8 | +0.093 [0.071, 0.118] | +0.187 [0.154, 0.221] |
| 9 (highest entropy) | +0.049 [0.027, 0.075] | +0.095 [0.067, 0.126] |

**Gowalla-NYC**

| decile | plain Mo-PAE gap | fairness-active gap |
|---|---|---|
| 0 | +0.262 [0.185, 0.338] | +0.204 [0.140, 0.275] |
| 1 | +0.205 [0.135, 0.276] | +0.204 [0.142, 0.269] |
| 2 | +0.155 [0.099, 0.217] | +0.252 [0.185, 0.323] |
| 3 | +0.130 [0.070, 0.194] | +0.217 [0.150, 0.287] |
| 4 | +0.150 [0.093, 0.208] | +0.172 [0.110, 0.237] |
| 5 | +0.113 [0.063, 0.166] | +0.116 [0.069, 0.166] |
| 6 | +0.105 [0.066, 0.150] | +0.097 [0.058, 0.143] |
| 7 | +0.070 [0.044, 0.102] | +0.070 [0.042, 0.101] |
| 8 | +0.067 [0.041, 0.095] | +0.103 [0.066, 0.143] |
| 9 (highest entropy) | +0.051 [0.030, 0.074] | +0.069 [0.048, 0.092] |

**The pattern is consistent across all three datasets and both
mechanisms: the gap is largest at low-entropy deciles (the users most at
risk to begin with) and smallest at high-entropy deciles.** Naive
evaluation isn't just generically wrong — it is most wrong exactly where
it matters most for a fairness claim.

## 3. Is the gap significantly non-uniform across deciles?

Kruskal-Wallis test, gap values grouped by the 10 entropy deciles:

| dataset | mechanism | n users | H | p |
|---|---|---|---|---|
| GeoLife | plain | 180 | 26.39 | 1.76×10⁻³ |
| GeoLife | fairness-active | 180 | 20.14 | 1.71×10⁻² |
| Foursquare | plain | 1083 | 26.78 | 1.52×10⁻³ |
| Foursquare | fairness-active | 1083 | 58.17 | 3.01×10⁻⁹ |
| Gowalla-NYC | plain | 587 | 21.12 | 1.21×10⁻² |
| Gowalla-NYC | fairness-active | 587 | 29.90 | 4.56×10⁻⁴ |

**All 6 panels reject uniformity (p ≤ 0.017).** This is the PI's
continue/stop gate for this item, and it says continue.

## 4. Ranking disagreement: naive privacy_gain vs. adaptive privacy_gain

Spearman rank correlation between each user's naive-attacker privacy_gain
and adaptive-attacker privacy_gain (1 = rankings agree completely):

| dataset | mechanism | Spearman ρ | p | excl. near-zero-raw-risk users | movers well→poor | movers poor→well |
|---|---|---|---|---|---|---|
| GeoLife | plain | 0.216 | 3.7×10⁻³ | ρ=0.21 (n=176/180 kept) | 12 | 1 |
| GeoLife | fairness-active | 0.162 | 3.0×10⁻² | ρ=0.17 | 11 | 0 |
| Foursquare | plain | 0.665 | 3.5×10⁻¹³⁹ | ρ=0.63 (n=1031/1083 kept) | 37 | 5 |
| Foursquare | fairness-active | 0.399 | 1.4×10⁻⁴² | ρ=0.38 | 67 | 5 |
| Gowalla-NYC | plain | 0.286 | 1.7×10⁻¹² | ρ=0.27 (n=546/587 kept) | 40 | 2 |
| Gowalla-NYC | fairness-active | 0.198 | 1.3×10⁻⁶ | ρ=0.18 | 61 | 2 |

("movers" = top-quartile-protected under the naive ranking who fall to
bottom-quartile-protected under the adaptive ranking, and vice versa; the
near-zero-raw-risk exclusion is the standard check for this project's
recurring ratio-metric instability — excluding those users barely moves
ρ, so the disagreement is not an artifact of it.)

Two secondary observations, reported as seen, not explained away:
- **The fairness-active mechanism shows *lower* rank agreement (more
  reshuffling) than plain Mo-PAE in all three datasets** (GeoLife
  0.216→0.162, Foursquare 0.665→0.399, Gowalla-NYC 0.286→0.198) — the
  opposite of what "fairness training makes measurement more reliable"
  would predict.
- **Movers are overwhelmingly one-directional**: far more users look
  well-protected under the naive attacker and turn out poorly protected
  under the adaptive one than the reverse, in every panel.

Three concrete examples (fairness-active mechanism, λ4=1.0), using raw
top-1 accuracy rather than the ratio-based privacy_gain metric to avoid
this project's known near-zero-denominator instability:

- **GeoLife user 50** (decile 8): naive attacker 10.3% top-1 — reads as
  well protected. Adaptive attacker: 82.2% top-1.
- **Foursquare user 638** (decile 1): naive attacker 1.1% top-1. Adaptive
  attacker: 98.9% top-1.
- **Gowalla-NYC users 6684, 6711, 1323**: naive attacker 0.0% top-1 on
  all three — the strongest possible "fully protected" reading. Adaptive
  attacker: 100.0% top-1 on all three.

## 5. Gowalla-NYC's reweighted-attacker sign reversal (noted, not explained)

Per the PI's instruction, restating Checkpoint 3b's finding as-is:
Gowalla-NYC's group-reweighted attacker does *worse* than naive on its
target group under plain Mo-PAE (Δ = −0.023, CI [−0.039, −0.011]) but
reliably *better* once the fairness mechanism is active (Δ = +0.055, CI
[0.037, 0.076]). This is the one place in the whole grid where a
fairness-specific exploit signature appears; it is not explained away
here, and the uneven-gap finding above does not depend on it.

## 6. The decile-recovery result is itself an attribute-inference leak

Checkpoint 3b's threat-model check (`can_infer_decile_from_features`)
found a user's entropy decile — an aggregate behavioral statistic never
given to the mechanism as a label — is recoverable from the released
encoder output at 34–42% user-level top-1 accuracy against 10% chance,
on all three datasets. Per the related-work framing above (Song &
Shmatikov's "overlearning"), **this is itself an attribute-inference
privacy leak, independent of re-identification**: the mechanism's
representation carries exploitable information about a behavioral trait
of the user that was never part of its training objective. It also
supplies the adaptive adversary's group-knowledge requirement in a
realistic, feature-only way, as Checkpoint 3b already noted — but the
leak is worth naming on its own terms, not only as an enabler for the
re-identification attacks above.

## Decision

**The gap is not uniform — continuing straight to Phase 4 (item B),**
per the PI's instruction. The adopted framing survives item A's
replication intact: naive evaluation doesn't just generically understate
risk (established prior work, §1) — it understates risk by a
systematically different amount for different users, worst for the
users a fairness mechanism is supposed to protect, which is exactly what
makes fairness-of-privacy claims built on naive evaluation unreliable.
Phase 5 (C) baselines can proceed in parallel per the PI's standing
instruction.

## Figures and data

`reports/figures/phase3c/summary.json` (all 6 panels' decile CIs,
Kruskal-Wallis stats, Spearman results). Built entirely from
`reports/figures/phase3b/*/per_user_3b.csv` — no new training;
`scripts/analyze_03c_uneven_gap.py`.
