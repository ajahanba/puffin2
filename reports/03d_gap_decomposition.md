# Checkpoint 3d — The adaptive-adversary gap, decomposed

**5-line summary:** The PI's challenge was correct. Decomposing the
naive-vs-specialized gap into candidate-set-size, specialized-training,
and side-information effects shows that **the "specialized attacker"
advantage reported since Checkpoint 3 is almost entirely explained by two
things that require no attacker sophistication at all**: restricting the
SAME naive model's candidate pool to the user's true entropy decile
(the "side information" component: positive and significant in every
decile, every dataset, +0.10 to +0.31) and oracle knowledge of that
decile. Actually *training* a specialized model on top of that
restriction adds nothing (the "specialization" component is at or below
zero almost everywhere). Worse: once the oracle assumption is replaced
with the ACTUAL decile classifier's predictions (34–42% accurate, per
Checkpoint 3b), the resulting "realistic" adaptive attacker
**underperforms the plain naive attacker** in 27 of 30 (dataset, decile)
combinations tested, CI excluding zero in most of them. **Per the PI's
explicit instruction, stopping and reporting here — not proceeding to
Phase 4 or paper drafting.**

## 0. What this corrects

Every adaptive-adversary number reported since Checkpoint 3
(`reports/03_attack_suite.md`'s original 2.4–2.8x, Checkpoint 3b's
multi-seed replication, Checkpoint 3c's uneven-gap analysis, and Phase
4's entire evaluation standard) used `per_group_specialized_attack`
exactly as the PI described: trained and tested within each user's true
decile (GeoLife: ~18 candidates instead of ~180 — a ~10x smaller
problem) with oracle access to that decile. None of those reports
distinguished how much of the result came from attacker sophistication
versus those two assumptions. This checkpoint does.

## 1. Per-user minimum (item 3)

Users with fewer than 10 test windows are excluded from every per-user
metric and example below, per the PI's instruction. Exclusion counts:

| dataset | n before filter | n excluded | % excluded |
|---|---|---|---|
| GeoLife | 181 | 3 | 1.7% |
| Foursquare | 1083 | 0 | 0.0% |
| Gowalla-NYC | 1517 | 1245 | **82.1%** |

Gowalla-NYC's exclusion rate is severe — 4 in 5 "users" in this dataset
have fewer than 10 test windows under this pipeline's SL=10 windowing,
and their per-user top-1 numbers (0% or 100% on 1–9 windows) were never
meaningful. Every Gowalla-NYC number in every table below is already
computed on the surviving ~18% (272 users); per-decile sample sizes are
shown explicitly because they're small and uneven (n=10 to n=83).
**This likely also affects every other Gowalla-NYC number reported in
Checkpoints 3/3b/3c/4** — none of them applied this filter. Revisiting
those with the filter applied is a natural follow-up, not done here
(out of this item's scope).

## 2. The five quantities, per decile, per dataset

(a) naive, (b) naive restricted to the true decile's candidates (same
model as (a), no retraining), (c) specialized (oracle-routed, train AND
test within the true decile), (d) reweighted (alone), (e) realistic
(specialized heads from (c), routed at test time by the actual decile
*classifier's prediction*, not the oracle label). All values are per-user
top-1 means, averaged over 5 attacker seeds, after the per-user minimum
filter.

**GeoLife** (lambda4=1.0, 178 users after filter)

| decile | a | b | c | d | e |
|---|---|---|---|---|---|
| 0 | 0.286 | 0.467 | 0.430 | 0.373 | 0.206 |
| 1 | 0.246 | 0.431 | 0.438 | 0.222 | 0.202 |
| 2 | 0.161 | 0.361 | 0.319 | 0.112 | 0.107 |
| 3 | 0.111 | 0.277 | 0.276 | 0.114 | 0.085 |
| 4 | 0.107 | 0.283 | 0.204 | 0.106 | 0.053 |
| 5 | 0.152 | 0.356 | 0.286 | 0.149 | 0.086 |
| 6 | 0.073 | 0.213 | 0.205 | 0.068 | 0.049 |
| 7 | 0.101 | 0.230 | 0.225 | 0.099 | 0.083 |
| 8 | 0.094 | 0.261 | 0.231 | 0.097 | 0.062 |
| 9 | 0.052 | 0.163 | 0.126 | 0.052 | 0.040 |

**Foursquare** (1083 users, 0 excluded)

| decile | a | b | c | d | e |
|---|---|---|---|---|---|
| 0 | 0.696 | 0.830 | 0.789 | 0.754 | 0.536 |
| 1 | 0.538 | 0.755 | 0.737 | 0.522 | 0.398 |
| 2 | 0.451 | 0.678 | 0.632 | 0.433 | 0.250 |
| 3 | 0.445 | 0.689 | 0.633 | 0.426 | 0.174 |
| 4 | 0.383 | 0.605 | 0.552 | 0.366 | 0.159 |
| 5 | 0.335 | 0.539 | 0.483 | 0.330 | 0.107 |
| 6 | 0.215 | 0.415 | 0.387 | 0.211 | 0.093 |
| 7 | 0.212 | 0.418 | 0.366 | 0.214 | 0.083 |
| 8 | 0.132 | 0.313 | 0.274 | 0.134 | 0.073 |
| 9 | 0.125 | 0.223 | 0.209 | 0.121 | 0.107 |

**Gowalla-NYC** (272 users after filter — see §1's caveat; n per decile in parens)

| decile (n) | a | b | c | d | e |
|---|---|---|---|---|---|
| 0 (10) | 0.560 | 0.731 | 0.686 | 0.614 | 0.526 |
| 1 (18) | 0.453 | 0.643 | 0.611 | 0.435 | 0.372 |
| 2 (11) | 0.496 | 0.749 | 0.748 | 0.479 | 0.343 |
| 3 (19) | 0.399 | 0.654 | 0.645 | 0.396 | 0.287 |
| 4 (15) | 0.221 | 0.483 | 0.476 | 0.202 | 0.057 |
| 5 (18) | 0.237 | 0.493 | 0.487 | 0.241 | 0.050 |
| 6 (20) | 0.274 | 0.585 | 0.562 | 0.276 | 0.090 |
| 7 (30) | 0.173 | 0.339 | 0.347 | 0.165 | 0.063 |
| 8 (48) | 0.147 | 0.299 | 0.260 | 0.150 | 0.064 |
| 9 (83) | 0.105 | 0.183 | 0.182 | 0.105 | 0.135 |

## 3. Gap components (bootstrap 95% CI, paired per user)

`side_information = b−a` (candidate-pool restriction alone, no
retraining), `specialization = c−b` (does training a dedicated model on
top of that restriction help further?), `realistic = e−a` (what a
real adversary without oracle group knowledge actually gets).

**GeoLife**

| decile | side_information | specialization | realistic |
|---|---|---|---|
| 0 | +0.181 [0.117,0.244] | −0.037 [−0.084,0.008] | **−0.080 [−0.168,−0.017]** |
| 1 | +0.186 [0.119,0.256] | +0.007 [−0.041,0.063] | **−0.044 [−0.081,−0.009]** |
| 2 | +0.200 [0.122,0.289] | −0.042 [−0.092,0.005] | **−0.054 [−0.111,−0.002]** |
| 3 | +0.166 [0.112,0.230] | −0.001 [−0.063,0.061] | −0.026 [−0.062,0.003] (CI incl. 0) |
| 4 | +0.176 [0.116,0.246] | **−0.079 [−0.130,−0.035]** | **−0.054 [−0.109,−0.009]** |
| 5 | +0.204 [0.148,0.264] | **−0.069 [−0.131,−0.014]** | **−0.066 [−0.103,−0.032]** |
| 6 | +0.140 [0.092,0.192] | −0.008 [−0.034,0.024] | **−0.023 [−0.041,−0.006]** |
| 7 | +0.129 [0.090,0.171] | −0.005 [−0.025,0.017] | **−0.017 [−0.030,−0.005]** |
| 8 | +0.168 [0.109,0.243] | −0.030 [−0.065,0.003] | **−0.032 [−0.046,−0.017]** |
| 9 | +0.111 [0.039,0.193] | **−0.036 [−0.067,−0.006]** | **−0.013 [−0.025,−0.000]** |

**Foursquare**

| decile | side_information | specialization | realistic |
|---|---|---|---|
| 0 | +0.134 [0.102,0.169] | **−0.041 [−0.067,−0.019]** | **−0.160 [−0.203,−0.115]** |
| 1 | +0.216 [0.179,0.256] | −0.018 [−0.038,0.001] | **−0.140 [−0.182,−0.101]** |
| 2 | +0.226 [0.186,0.267] | **−0.045 [−0.070,−0.021]** | **−0.201 [−0.245,−0.159]** |
| 3 | +0.244 [0.210,0.279] | **−0.056 [−0.077,−0.036]** | **−0.270 [−0.319,−0.224]** |
| 4 | +0.222 [0.188,0.256] | **−0.054 [−0.079,−0.032]** | **−0.224 [−0.269,−0.183]** |
| 5 | +0.204 [0.173,0.240] | **−0.056 [−0.078,−0.034]** | **−0.229 [−0.280,−0.177]** |
| 6 | +0.201 [0.172,0.231] | **−0.028 [−0.051,−0.006]** | **−0.121 [−0.161,−0.086]** |
| 7 | +0.206 [0.175,0.243] | **−0.052 [−0.079,−0.027]** | **−0.129 [−0.168,−0.094]** |
| 8 | +0.181 [0.152,0.211] | **−0.039 [−0.064,−0.017]** | **−0.059 [−0.088,−0.033]** |
| 9 | +0.098 [0.077,0.121] | −0.014 [−0.035,0.006] | −0.018 [−0.038,0.000] (CI incl. 0) |

**Gowalla-NYC** (n per decile as in §2's table — small, uneven)

| decile | side_information | specialization | realistic |
|---|---|---|---|
| 0 | +0.171 [0.075,0.271] | −0.045 [−0.126,0.030] | −0.034 [−0.101,0.027] (CI incl. 0) |
| 1 | +0.189 [0.121,0.273] | −0.032 [−0.131,0.055] | **−0.082 [−0.156,−0.010]** |
| 2 | +0.254 [0.155,0.364] | −0.001 [−0.092,0.087] | **−0.152 [−0.279,−0.046]** |
| 3 | +0.255 [0.180,0.341] | −0.009 [−0.055,0.035] | **−0.113 [−0.197,−0.035]** |
| 4 | +0.262 [0.169,0.371] | −0.007 [−0.090,0.078] | **−0.165 [−0.301,−0.048]** |
| 5 | +0.257 [0.135,0.388] | −0.006 [−0.077,0.060] | **−0.187 [−0.309,−0.080]** |
| 6 | +0.311 [0.218,0.401] | −0.023 [−0.106,0.046] | **−0.184 [−0.274,−0.091]** |
| 7 | +0.167 [0.113,0.229] | +0.008 [−0.057,0.063] | **−0.109 [−0.190,−0.038]** |
| 8 | +0.152 [0.113,0.191] | **−0.039 [−0.072,−0.012]** | **−0.083 [−0.133,−0.038]** |
| 9 | +0.078 [0.061,0.097] | −0.001 [−0.016,0.015] | **+0.030 [0.015,0.045]** |

**Across all 30 (dataset, decile) cells: side_information's CI excludes
zero (positive) in all 30. specialization's CI excludes zero in 9 of 30,
every one of those 9 negative. realistic's CI excludes zero in 27 of 30
— 26 negative, and Gowalla-NYC's decile 9 the lone positive exception,
reported as found, not explained away** (consistent with this project's
practice for Gowalla-NYC's reweighted-attacker sign reversal in
Checkpoint 3b).

## 4. Correcting Checkpoint 3c's claim (item 2)

`reports/03c_uneven_gap.md` said the measurement gap (there, computed as
`max(specialized, reweighted) − naive`, oracle-routed) was "largest at
low-entropy deciles... smallest at high-entropy deciles" — a claim the
PI correctly flagged as not supported by its own tables: GeoLife-plain's
decile 0 was +0.034 with a CI that includes zero (not even significant),
and Foursquare's gap peaks in deciles 1–4, not decile 0.

**Corrected statement**: the gap is non-uniform across deciles, and
**smallest at the highest-entropy decile** — not simply "largest at the
lowest." This holds up on the decomposed `realistic` gap too: in all
three datasets, decile 9 has the smallest-magnitude `realistic` value
(GeoLife −0.013, Foursquare −0.018, Gowalla-NYC +0.030 — the
near-zero/flipped case), while the largest-magnitude decile is
dataset-specific, not always decile 0 (GeoLife: decile 0 and 5 roughly
tie around −0.08; Foursquare: decile 3, −0.270; Gowalla-NYC: decile 5-6,
around −0.18 to −0.19). The original 3c claim's error was reading a
noisy, non-monotonic shape as a clean monotonic one.

## 5. What this means

The specialized/reweighted attacker's apparent 2.4–2.8x advantage
(Checkpoint 3), its apparent persistence under plain Mo-PAE (Checkpoint
3b), and its use as Phase 4's entire evaluation standard were all built
on an attacker that gets a ~10x easier problem (restricted candidate
pool) AND knows the answer to a question (true entropy decile) it would
otherwise have to guess at ~35-40% accuracy. Component (b) alone (same
model, smaller candidate pool, no retraining) explains nearly the whole
originally-reported gap. Component (c) (actually training a specialized
model) adds nothing beyond that in most deciles, and significantly hurts
in several. Component (e) (what's left once oracle knowledge is removed)
is *negative* almost everywhere — a realistic adaptive adversary,
built exactly as this project built it, is not more dangerous than a
naive one; if anything, it's slightly worse, likely because splitting
training data across heads and routing test data through a noisy
classifier loses more than the group-specialization gains.

This does not mean Mo-PAE's fairness mechanism is fine, or that the
earlier checkpoints' other findings (disparate risk in Phase 2, the
uneven naive-attacker *measurement error* reframing from Checkpoint 3c
once separated from the retracted "specialized attacker" framing, the
decile-inference attribute leak) are wrong — those don't depend on the
specialized/reweighted attacker's validity. But the specific "adaptive
adversary recovers more risk" claim that has anchored Checkpoints 3, 3b,
3c's title, and all of Phase 4's evaluation standard does not survive
this decomposition as stated.

**Per the PI's explicit instruction: stopping and reporting here. Not
proceeding to Phase 4 (item 4) or paper drafting** without the PI's
review of this correction and a decision on how — or whether — to
re-ground the adaptive-adversary standard going forward (e.g., on
component (b)'s candidate-restriction effect alone, explicitly labeled
as not requiring attacker sophistication, or on component (e)'s
realistic routing as the honest "adaptive adversary" metric, which would
substantially change what Phase 4 is supposed to defend against).

## Figures and data

`reports/figures/phase3d/<dataset>/{summary.json,per_user_3d.csv,excluded_users.csv}`
for all 3 datasets. Driver: `scripts/run_phase3d_gap_decomposition.py`.
Core module: `puffin/attacks/gap_decomposition.py`, tested
(`tests/test_gap_decomposition.py`).
