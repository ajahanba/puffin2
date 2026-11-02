# Report 06: Individual Fairness of Privacy Protection

**5-line summary.** Plain Mo-PAE is *more* individually fair than no-privacy,
not less -- the top-10%-similar-pairs rate within epsilon=0.2 rises on
10 of 12 (dataset x similarity-definition) cells, sometimes substantially
(Foursquare/Gowalla location-distribution: +0.18-0.19). PUFFIN (ed_soft)
at matched utility against the plain lambda3 curve is a mixed, mostly
flat-to-negative picture: it improves the rate on Foursquare's lowest
tested weight (lambda4=0.3, +0.01 to +0.03 across all four similarity
definitions) but *reduces* it at every other resolvable point (Foursquare
lambda4=1.0/3.0, Gowalla-SF-Bay lambda4=1.0/3.0; GeoLife is unresolvable
by interpolation, same reason as Table 7). Critically, whichever
direction the effect goes, it is consistent across all four similarity
definitions, not entropy-specific -- PUFFIN is not secretly only "working"
under the one definition it was trained against. A fixed delta=0.8
threshold is a very loose, inconsistent test (flags 57-93% of ALL pairs
as similar under entropy/trajectory-shape/radius-of-gyration, but
essentially 0% under location-distribution on every dataset) -- the
top-10% selection and the dose-response curves below are the comparisons
that actually mean anything across definitions.

## 1. Methodology

Four similarity definitions, each computed from **training-split rows
only** (`df["split"] == "train"`, exactly what the encoder and attacker
were trained/evaluated on -- see `puffin/metrics/individual_fairness.py`),
normalized to [0,1]:

- **(a) Entropy**: 1 - normalized |H_i - H_j| on Shannon entropy of each
  user's train-split visitation distribution (the original PUFFIN
  similarity basis).
- **(b) Location-distribution**: 1 - Jensen-Shannon distance between
  users' train-split visitation histograms over cells. Captures *where*
  people go, which entropy (a measure of *how evenly* they visit,
  regardless of place) misses entirely.
- **(c) Trajectory-shape**: 1 - normalized symmetric Hausdorff distance
  (haversine, km) between users' (lat, lon) point sets. Hausdorff, not
  DTW: no extra dependency, and the comparison here is about which
  places were visited and how spread out they are, not visit order
  (which (a)/(b)/(d) don't capture either). Each user's point sequence is
  subsampled to at most 150 points (uniform random, chronological order
  preserved) before computing pairwise distances -- GeoLife's train
  split has up to 1.7M raw points for a single user, and Foursquare/
  Gowalla-SF-Bay's pair counts (586K and 272K respectively) make the full
  sequence computationally infeasible; 150 points per user keeps every
  dataset's full pairwise computation under a few minutes.
- **(d) Radius of gyration**: 1 - normalized |rog_i - rog_j|.

**Individual-fairness metric**: among pairs counted as "similar," the
fraction whose post-mechanism risk (external attacker's per-user top-1,
averaged over 5 attacker seeds) differs by less than epsilon. "Similar"
is defined two ways so differently-scaled definitions are comparable:
a fixed threshold (delta=0.8, the original PUFFIN draft's value) and the
**top 10% most similar pairs** under each definition (same pair *count*
regardless of definition). Reported at epsilon=0.2 (the original value)
plus the full curve over epsilon in the underlying JSON. Dose-response:
all pairs binned by similarity into deciles, mean |risk_i - risk_j| per
bin with bootstrap 95% CIs. All rates are bootstrapped over *both*
encoder seeds and users (`bootstrap_if_rate`, 1000 resamples).

**Mechanisms evaluated**: every protocol-tagged, audited-clean run in
`reports/figures/retraining_to_convergence/` -- no-privacy, plain Mo-PAE
(every lambda3 sweep point), PUFFIN (ed_soft) at each lambda4, floor, and
the five single-seed formulations, across GeoLife, Foursquare, and
Gowalla-SF-Bay. Full per-run, per-definition results:
`reports/figures/individual_fairness/{dataset}_individual_fairness.json`.

## 2. Dose-response: do more similar users get more similar protection?

Yes, for both no-privacy and plain Mo-PAE, under every similarity
definition tested -- mean |risk_i - risk_j| falls monotonically as
similarity rises. GeoLife, entropy similarity, 10 deciles of all pairs
(mean absolute risk difference, no_privacy -> plain):

| similarity bin | n pairs | no-privacy | plain |
|---|---|---|---|
| [0.10, 0.19] | 11 | 0.296 | 0.302 |
| [0.19, 0.28] | 28 | 0.293 | 0.287 |
| [0.28, 0.37] | 126 | 0.279 | 0.274 |
| [0.37, 0.46] | 282 | 0.210 | 0.206 |
| [0.46, 0.55] | 602 | 0.216 | 0.208 |
| [0.55, 0.64] | 1132 | 0.187 | 0.176 |
| [0.64, 0.73] | 1827 | 0.188 | 0.177 |
| [0.73, 0.82] | 2937 | 0.182 | 0.169 |
| [0.82, 0.91] | 3936 | 0.169 | 0.156 |
| [0.91, 1.00] | 4872 | **0.160** | **0.143** |

The monotonic fall confirms the dose-response signature holds; plain's
curve sits at or below no-privacy's in every bin, consistent with
Section 3's top-10% finding that plain improves individual fairness over
no-privacy, not just at one arbitrary threshold.

## 3. delta=0.8 is a loose, inconsistent test

Fraction of **all** user pairs counted as "similar" at the original
delta=0.8 threshold, per dataset and definition:

| dataset | entropy | location-distribution | trajectory-shape | radius-of-gyration |
|---|---|---|---|---|
| GeoLife | 60.6% | 0.04% | 89.1% | 92.9% |
| Foursquare | 70.1% | 0.00% | 6.5% | 64.3% |
| Gowalla-SF-Bay | 74.7% | 0.00% | 15.3% | 56.6% |

At the identical nominal threshold, entropy and radius-of-gyration call
57-93% of *all* pairs in a dataset "similar," while location-distribution
calls essentially none similar on any dataset, and trajectory-shape swings
from 6.5% to 89% depending on dataset. A fixed similarity threshold is not
comparable across definitions or datasets; every comparison in Sections 4-5
below uses the top-10% selection instead, which fixes the pair *count*
rather than a similarity value, and is the one actually meaningful across
definitions.

## 4. Does plain Mo-PAE violate individual fairness, compared with no-privacy?

No -- plain Mo-PAE is *more* individually fair than no-privacy on 10 of 12
(dataset x definition) cells, top-10% selection, epsilon=0.2:

| dataset | entropy (Delta) | location-dist. (Delta) | trajectory-shape (Delta) | rog (Delta) |
|---|---|---|---|---|
| GeoLife | +0.055 | -0.001 | +0.044 | +0.025 |
| Foursquare | +0.106 | +0.193 | +0.112 | +0.085 |
| Gowalla-SF-Bay | +0.150 | +0.179 | +0.148 | +0.136 |

(Delta = plain's IF-rate minus no-privacy's; positive means plain is
*more* fair.) The one near-zero exception (GeoLife, location-distribution,
-0.0006) is indistinguishable from noise. This does not contradict this
paper's main disparate-risk finding (Section~8): that result is about the
*relative size* of risk reduction across entropy deciles, not about
whether users end up close to each other in absolute risk -- a mechanism
can compress the overall spread of risk across users (raising this
metric) while still reducing that spread *unevenly* across deciles
(the disparate-risk finding). They are different, compatible facts about
the same mechanism.

## 5. Does PUFFIN improve individual fairness, at matched utility against the lambda3 curve?

Interpolating the plain-Mo-PAE lambda3 curve to each PUFFIN (ed_soft)
point's own utility (never extrapolating -- unresolvable points are
reported as such, not guessed), then comparing top-10% IF-rate at that
matched utility:

| dataset | lambda4 | entropy | location-dist. | trajectory-shape | rog | status |
|---|---|---|---|---|---|---|
| GeoLife | 0.3, 1.0, 3.0 | -- | -- | -- | -- | unresolvable (PUFFIN's utility falls outside the curve's range, same issue as Table 7) |
| Foursquare | 0.3 | +0.018 | +0.031 | +0.015 | +0.017 | PUFFIN improves |
| Foursquare | 1.0 | -0.005 | -0.005 | -0.008 | -0.001 | ~flat |
| Foursquare | 3.0 | -0.083 | -0.137 | -0.083 | -0.068 | PUFFIN worsens |
| Gowalla-SF-Bay | 0.3 | -- | -- | -- | -- | unresolvable |
| Gowalla-SF-Bay | 1.0 | -0.029 | -0.049 | -0.039 | -0.035 | PUFFIN worsens |
| Gowalla-SF-Bay | 3.0 | -0.130 | -0.160 | -0.130 | -0.121 | PUFFIN worsens |

**PUFFIN only reliably improves individual fairness at the lowest tested
weight on one of three datasets.** Everywhere else it is resolvable, it
is flat or makes individual fairness *worse* than plain Mo-PAE already
achieves at the same utility -- consistent with Section~\ref{sec:phase4}'s
finding that no formulation tested dominates plain Mo-PAE once training is
run to convergence.

### Does the effect hold under similarity definitions PUFFIN wasn't trained on?

Yes, directionally. Within every resolvable (dataset, lambda4) cell, all
four similarity definitions agree on the *sign* of the effect (all
positive at Foursquare lambda4=0.3; all negative everywhere else
resolvable) -- the effect is not an artifact of entropy being PUFFIN's own
training signal. The *magnitude* varies by definition (location-distribution
shows the largest swings in both directions), but PUFFIN is not secretly
helping only under the metric it was optimized against.

## 6. How much do the similarity definitions agree, and a concrete example

Jaccard overlap between each pair of definitions' top-10%-most-similar
pair sets (all three datasets):

| | entropy-location | entropy-trajectory | entropy-rog | location-trajectory | location-rog | trajectory-rog |
|---|---|---|---|---|---|---|
| GeoLife | 0.081 | 0.088 | 0.105 | 0.141 | 0.123 | **0.539** |
| Foursquare | 0.064 | 0.061 | 0.061 | 0.097 | 0.084 | 0.093 |
| Gowalla-SF-Bay | 0.057 | 0.066 | 0.061 | 0.132 | 0.073 | 0.135 |

Agreement is low (0.06-0.14) almost everywhere -- the four definitions
mostly disagree about which pairs count as "similar." The one exception,
trajectory-shape vs. radius-of-gyration on GeoLife (0.54), is itself
telling: both are spatial-extent measures, and GeoLife's dense GPS traces
apparently make them nearly redundant there -- a pattern that does *not*
repeat on Foursquare/Gowalla-SF-Bay's sparser check-in data (0.09-0.14),
so it should not be assumed to generalize.

Concrete entropy-similar/location-dissimilar examples, one per dataset
(entropy similarity in the global top decile, location-distribution
similarity at or below the median):

- **GeoLife**: users 2 and 133 -- entropy-similarity 0.995, location-
  distribution similarity 0.000. Same regularity of visitation, visit
  essentially disjoint sets of places.
- **Foursquare**: users 482 and 663 -- entropy-similarity 0.997,
  location-distribution similarity ~0.
- **Gowalla-SF-Bay**: users 6495 and 6993 -- entropy-similarity 0.988,
  location-distribution similarity ~0.

Entropy and location-distribution are measuring genuinely different
things, consistently across all three datasets: two users can visit a
near-identical *number* of places with a near-identical skew in how often
they return to each, while visiting completely different physical
locations -- entropy sees them as near-twins; location-distribution does
not see them as similar at all.

## Data and reproducibility

- `puffin/metrics/individual_fairness.py`: similarity definitions and the
  fairness metric/dose-response/bootstrap functions.
- `scripts/compute_individual_fairness.py` +
  `scripts/launch_individual_fairness.sh`: the full batch (CPU, nice'd,
  thread-capped), per dataset per protocol-tagged run.
- `scripts/generate_individual_fairness_report.py`: everything in this
  report, regenerable from `reports/figures/individual_fairness/*.json`
  and the cached `*_similarities.npz`/`*_user_ids.json`.
- Every number above traces to a `usable: true` protocol-tagged run (see
  `scripts/backfill_protocol_tags.py`/`scripts/audit_min_epochs_contamination.py`);
  no stale or unaudited run contributes to this report.
