# Report 06: Individual Fairness of Privacy Protection

**5-line summary (revised).** The absolute epsilon=0.2 individual-fairness
(IF) rate mostly tracks overall privacy level, not genuine individual
fairness: on Foursquare, plain Mo-PAE at lambda3=0.5 has utility 0.060,
mean risk 0.030, and a top-10%-similar-pairs IF-rate of 0.98 -- near-total
"fairness" that is largely just "almost everyone's risk is near zero."
A scale-free measure (mean |risk_i-risk_j| over top-10%-similar pairs,
divided by the same quantity over ALL pairs -- lower means similar users
are treated more alike *relative to the population's own spread*, not
just because everyone's risk shrank together) tells a more mixed story:
on GeoLife, plain Mo-PAE's apparent fairness gain over no-privacy mostly
*disappears* once rescaled (3 of 4 definitions turn slightly negative);
on Foursquare and Gowalla-SF-Bay, a real, smaller effect survives for
entropy/location-distribution/trajectory-shape, but radius-of-gyration
shows essentially none anywhere. PUFFIN (ed_soft) at matched utility
against the plain lambda3 curve, redone with the scale-free measure, is
mixed on Foursquare/Gowalla-SF-Bay (modest improvement at the lowest
tested weight, worse at the highest) -- so that part of the earlier
conclusion survives the correction, even though Q1's headline claim does
not survive it cleanly everywhere. **GeoLife is now resolvable** (the
extended lambda3 sweep covers utility 0.46-0.76, finally bracketing
PUFFIN's own range) and gives a genuinely surprising, carefully-stated
result: at matched utility, ed_soft is LESS fair than plain's own curve
by the entropy-based scale-free measure (the exact similarity it was
trained on), but MORE fair by trajectory-shape -- while, by the
decile-level measure, ed_soft clearly beats plain's curve on decile-0
risk and reaches a decile-0 risk level plain cannot reach at any tested
privacy weight. At matched MEAN RISK instead of matched utility, PUFFIN
(lambda4=1.0) beats plain's most extreme tested point (lambda3=3) on
BOTH utility (0.723 vs. 0.501) and decile-0 risk (0.207 vs. 0.270)
simultaneously, at a modest cost to the already-best-off decile-9.

## 1. Methodology

(Similarity definitions, IF-rate/dose-response/bootstrap machinery
unchanged from the first version of this report -- see
`puffin/metrics/individual_fairness.py`. This revision adds:)

**Scale-free IF measure** (`puffin.metrics.individual_fairness.scale_free_if`):

```
ratio = mean(|risk_i - risk_j|  over the top-10%-most-similar pairs)
        -----------------------------------------------------------
        mean(|risk_i - risk_j|  over ALL pairs)
```

Lower ratio means risk differences among similar pairs are small
*relative to how much risk varies across the whole population*, which is
what "individual fairness" is actually supposed to mean. A mechanism that
simply compresses everyone's risk toward zero shrinks the numerator and
the denominator together and leaves the ratio roughly unchanged -- unlike
the raw absolute IF-rate, which such a mechanism can push toward 1 (near-
total "fairness") for free, with no real individual-fairness content.
Computed per dataset, per run, per similarity definition, with bootstrap
95% CIs over both encoder seeds and pairs (1000 resamples for the main
report; 300 for the matched-utility curve interpolation, for speed).
Full table: `reports/figures/individual_fairness/scale_free_if_table.json`.
Mean risk (plain average post-mechanism risk across all users) is
reported alongside every ratio, so the confound is visible rather than
hidden.

## 2. delta=0.8 is a loose, inconsistent test

(Unchanged -- this finding is about how differently-scaled similarity
definitions disagree about what counts as "similar," not about the
privacy-level confound, and is unaffected by it.)

| dataset | entropy | location-distribution | trajectory-shape | radius-of-gyration |
|---|---|---|---|---|
| GeoLife | 60.6% | 0.04% | 89.1% | 92.9% |
| Foursquare | 70.1% | 0.00% | 6.5% | 64.3% |
| Gowalla-SF-Bay | 74.7% | 0.00% | 15.3% | 56.6% |

## 3. The confound, demonstrated directly

Foursquare, plain Mo-PAE, lambda3=0.5: utility 0.060, mean risk 0.030,
raw top-10%-similar-pairs IF-rate (epsilon=0.2) = **0.98** under every
one of the four similarity definitions. The scale-free ratio for this
exact same run tells a very different story:

| similarity | raw IF-rate | scale-free ratio | mean risk |
|---|---|---|---|
| entropy | 0.98 | **0.624** [0.625, 0.656] | 0.030 |
| location-distribution | 0.98 | **0.453** [0.426, 0.505] | 0.030 |
| trajectory-shape | 0.98 | **0.885** [0.752, 0.989] | 0.030 |
| radius-of-gyration | 0.98 | **0.938** [0.908, 0.978] | 0.030 |

The raw rate made it look like this operating point achieves near-
perfect fairness under all four definitions equally. The scale-free
ratio shows real, substantial signal for entropy and especially
location-distribution (similar pairs differ by 45-62% of what random
pairs differ by), and almost none for trajectory-shape/radius-of-
gyration (88-94% of the random-pair difference -- barely better than
chance grouping). This single example is why every comparison below
uses the scale-free ratio as the primary number, with the raw rate and
mean risk kept for context, not as the headline.

## 4. Does plain Mo-PAE violate individual fairness, compared with no-privacy?

Scale-free ratio, no-privacy vs. plain Mo-PAE (lambda3=0.3), top-10%
selection. Delta = plain's ratio minus no-privacy's ratio; **negative
means plain is more fair** (lower ratio), positive means plain is less
fair:

| dataset | mean risk (no-priv -> plain) | entropy (Delta) | location-dist. (Delta) | trajectory-shape (Delta) | rog (Delta) |
|---|---|---|---|---|---|
| GeoLife | 0.142 -> 0.133 | -0.050 | **+0.029** | +0.016 | +0.006 |
| Foursquare | 0.423 -> 0.298 | -0.052 | -0.181 | -0.045 | -0.002 |
| Gowalla-SF-Bay | 0.274 -> 0.188 | -0.051 | -0.072 | -0.035 | +0.003 |

**This does not hold up cleanly on GeoLife.** 3 of 4 definitions turn
*positive* (plain slightly less fair than no-privacy) once rescaled --
the raw-rate version of this report claimed a uniform improvement on
GeoLife (all four definitions positive in the old rate-based framing);
that claim was substantially an artifact of GeoLife's mean risk falling
from 0.142 to 0.133, not a genuine fairness gain. **On Foursquare and
Gowalla-SF-Bay, a real effect survives** for entropy, location-
distribution, and trajectory-shape (all negative, i.e. genuinely more
fair) even after controlling for the drop in mean risk -- location-
distribution shows the largest genuine effect on both (-0.181, -0.072).
**Radius-of-gyration shows essentially no genuine fairness signal on any
dataset** (+0.006, -0.002, +0.003 -- all indistinguishable from zero in
practical terms): whatever apparent "fairness" radius-of-gyration showed
under the raw rate was almost entirely the privacy-level confound.

## 5. Does PUFFIN improve individual fairness, at matched utility against the lambda3 curve?

**GeoLife is now resolvable.** The extended lambda3 sweep ({0.02,...,0.8,
1.2,2,3}, lambda3=2 and 3 using their plateau-audit-corrected data -- see
Section 9) covers utility 0.46-0.76, bracketing all three PUFFIN points
(0.71-0.74) for the first time; 0/5000 bootstrap draws require
extrapolation at every lambda4. Scale-free ratio, PUFFIN (ed_soft) vs.
the plain lambda3 curve interpolated to PUFFIN's own utility. Delta =
ed_soft's ratio minus the curve's interpolated ratio; **negative means
PUFFIN is more fair** at that matched utility:

| dataset | lambda4 | entropy | location-dist. | trajectory-shape | rog | status |
|---|---|---|---|---|---|---|
| GeoLife | 0.3 | **+0.089** | -0.004 | **-0.119** | -0.066 | mixed: worse on its OWN training signal, better on 2 of 3 others |
| GeoLife | 1.0 | **+0.095** | -0.002 | **-0.119** | -0.059 | same pattern |
| GeoLife | 3.0 | **+0.100** | +0.003 | **-0.107** | -0.058 | same pattern |
| Foursquare | 0.3 | -0.0095 | **-0.033** | -0.0057 | -0.0066 | PUFFIN more fair, all 4 |
| Foursquare | 1.0 | -0.0008 | -0.0018 | +0.0082 | +0.0002 | ~flat, mixed sign |
| Foursquare | 3.0 | +0.0296 | **+0.0975** | +0.0209 | -0.0010 | PUFFIN less fair, 3 of 4 |
| Gowalla-SF-Bay | 0.3 | -- | -- | -- | -- | unresolvable |
| Gowalla-SF-Bay | 1.0 | -0.0058 | +0.0606 | +0.0276 | -0.0016 | mixed, 2 of 4 worse |
| Gowalla-SF-Bay | 3.0 | +0.0410 | **+0.0942** | +0.0320 | +0.0023 | PUFFIN less fair, all 4 |

**GeoLife's result is genuinely surprising and worth stating carefully.**
At matched utility, ed_soft is *less* scale-free-fair than plain's own
curve under the exact similarity basis it was trained on (entropy,
consistently worse by +0.09 to +0.10 across all three lambda4) -- but
*more* fair under trajectory-shape (consistently better by -0.11 to
-0.12) and, more weakly, radius-of-gyration. Location-distribution is
essentially a wash. This does not contradict Section 4's decile-level
finding (ed_soft lowers decile-0 risk substantially at matched utility --
see the frontier test below): a mechanism can genuinely compress the
worst-off decile's risk while, by the entropy-based scale-free measure
specifically, not bringing entropy-similar pairs any closer together
than a same-utility point on plain's own curve already would. The two
measures are asking different questions (decile-level risk reduction vs.
whether similar-under-this-definition pairs end up close in risk), and
GeoLife is the one dataset where they disagree in sign.

**The decile-level frontier test (Table 7's analysis, now fully resolved)**
confirms the positive reading: at matched utility, ed_soft beats plain's
curve by +0.111 to +0.132 on decile-0 risk and +0.022 to +0.039 on mean
risk over all users (both significant, 0/5000 extrapolated), at a cost of
-0.017 to -0.022 on decile-9 risk (ed_soft worse there) -- an order of
magnitude smaller than the decile-0 gain. Plain's curve's own best-ever
decile-0 risk, reached only at its most extreme tested point (lambda3=3,
utility 0.50), is 0.270; PUFFIN reaches 0.200-0.226 at every lambda4
tested, at utility 0.71-0.74 -- a decile-0 risk level plain cannot reach
at any privacy weight tested, now confirmed with a properly powered,
non-extrapolated comparison.

**Matched mean risk (the complementary question: at the same overall
privacy level, which gives better utility and better worst-decile
protection?).** PUFFIN lambda4=0.3's mean risk (0.110) falls inside the
curve's range: at that matched mean risk, PUFFIN beats the curve's
interpolated point on BOTH utility (+0.139: 0.735 vs. 0.597) and
decile-0 risk (-0.077: 0.225 vs. 0.303). lambda4=1.0's own mean risk
(0.0962) sits 0.0003 below the curve's single lowest point (lambda3=3,
corrected, mean risk 0.0965) -- too close to interpolate honestly, but
close enough to compare directly instead:

| | utility | decile-0 risk | decile-9 risk | mean risk |
|---|---|---|---|---|
| PUFFIN (lambda4=1.0) | 0.723 | 0.207 | 0.049 | 0.0962 |
| plain Mo-PAE (lambda3=3, corrected) | 0.501 | 0.270 | 0.023 | 0.0965 |

At essentially the same overall privacy level, PUFFIN gives *much* higher
utility (+0.22) and lower decile-0 risk (-0.063) than plain's most extreme
tested point, at the cost of the already-best-off decile-9 being somewhat
worse off (+0.026) -- PUFFIN is not just moving along the same
utility-risk trade-off plain already offers; it reaches a better combined
operating point for the worst-off decile specifically. (Plain lambda3=3's
utility here, 0.501, is the plateau-audit-corrected value -- an earlier,
pre-audit reading of this same point gave 0.460; see Section 9.)

## 6. How much do the similarity definitions agree, and a concrete example

(Unchanged -- not affected by the privacy-level confound, since Jaccard
overlap between top-10% *sets* does not depend on the risk metric at
all.)

| | entropy-location | entropy-trajectory | entropy-rog | location-trajectory | location-rog | trajectory-rog |
|---|---|---|---|---|---|---|
| GeoLife | 0.081 | 0.088 | 0.105 | 0.141 | 0.123 | **0.539** |
| Foursquare | 0.064 | 0.061 | 0.061 | 0.097 | 0.084 | 0.093 |
| Gowalla-SF-Bay | 0.057 | 0.066 | 0.061 | 0.132 | 0.073 | 0.135 |

Concrete entropy-similar/location-dissimilar examples (entropy
similarity in the global top decile, location-distribution similarity at
or below the median):

- **GeoLife**: users 2 and 133 -- entropy-similarity 0.995, location-
  distribution similarity 0.000.
- **Foursquare**: users 482 and 663 -- entropy-similarity 0.997,
  location-distribution similarity ~0.
- **Gowalla-SF-Bay**: users 6495 and 6993 -- entropy-similarity 0.988,
  location-distribution similarity ~0.

## 7. Dose-response (unaffected by the confound's direction, same caveat applies to magnitude)

GeoLife, entropy similarity, mean absolute risk difference by similarity
decile, no-privacy vs. plain -- still monotonically decreasing for both,
confirming the dose-response signature holds. (Table unchanged from the
first version of this report; interpreting the *size* of the gap between
the two curves is subject to the same mean-risk confound as Section 4,
since this is still an absolute, not scale-free, quantity.)

## 8. Task 2, as it lands (key evidence for RQ1)

Still training (Lipschitz, both similarity bases, 3 datasets; PUFFIN-
with-location-distribution-similarity on GeoLife). The same measures as
above, applied to every run as it completes:

**GeoLife, PUFFIN trained on location-distribution similarity instead of
entropy** (2 of 2 lambda4 values landed so far; raw top-10% rate for
context, scale-free ratio is the number that matters, mean risk is the
same for both runs since similarity-basis doesn't change the trained
mechanism's risk, only which similarity it was optimized against):

| lambda4 | mean risk | entropy ratio | location-dist. ratio | trajectory-shape ratio | rog ratio |
|---|---|---|---|---|---|
| 0.3 | 0.133 | 0.812 [0.767,0.891] | 0.726 [0.655,0.881] | **1.320** [1.242,1.429] | **1.223** [1.150,1.301] |
| 1.0 | 0.130 | 0.817 [0.765,0.906] | 0.731 [0.652,0.847] | **1.322** [1.231,1.446] | **1.219** [1.127,1.306] |

**A genuinely new and important result: training PUFFIN against
location-distribution similarity helps under the similarity it was
trained on (and, less strongly, under entropy), but makes trajectory-
shape and radius-of-gyration's individual fairness actively *worse than
random grouping*** -- a ratio above 1 means pairs the mechanism treats as
"similar" under those two definitions end up with *more* different risk
than a random pair would, not less. This is a stronger and more concrete
version of the entropy-trained PUFFIN's Section 5 finding (improvement
does not generalize across similarity bases): here, non-generalization
is not just "no effect," it is a measurable reversal for two of the four
definitions. Lipschitz (both similarity bases, all three datasets) will
be added here as each lands, including the matched-utility-against-the-
plain-curve version of this table once enough of the curve's own points
are available under the same protocol.

## 9. Plateau-vs-genuine-collapse audit (new infrastructure, applied so far to the GeoLife lambda3 sweep)

Per the PI's instruction (2026-11-02): at higher privacy weights, a seed
sitting at a low-utility plateau may be a genuine collapse (the privacy
weight overwhelming prediction), not the known cold-start artifact --
the two cannot be told apart by epoch count alone. `puffin/eval/
plateau_audit.py` flags any seed below half its point's own median
utility, reruns it with `min_epochs` doubled, and classifies
`artifact_fixed` vs. `genuine_collapse` by whether it still sits at the
floor.

Applied to all four extended-sweep points. lambda3=1.2: 0 flagged,
clean. lambda3=2.0: 1 flagged (seed 3, original utility 0.0622 at 20
epochs; rerun at min_epochs=40 reached 0.6653 at 80 epochs --
**artifact_fixed**). lambda3=3.0: 1 flagged (seed 6, original utility
0.1465 at 20 epochs; rerun at min_epochs=40 reached 0.5625 at 80 epochs
-- **artifact_fixed**, consistent with its siblings' 0.37-0.61 range).
Neither flagged seed turned out to be a genuine collapse. Per-seed
utilities and the audited mean/median for every lambda3 point:

| lambda3 | per-seed utilities | naive mean | naive median | audited mean | audited median | genuine collapses |
|---|---|---|---|---|---|---|
| 1.2 | 0.676, 0.715, 0.724, 0.750, 0.730, 0.717, 0.737, 0.715, 0.719, 0.728 | 0.721 | 0.722 | 0.721 | 0.722 | 0 |
| 2.0 | 0.645, 0.628, 0.667, **0.665\***, 0.571, 0.682, 0.629, 0.621, 0.683, 0.645 | 0.583 | 0.637 | 0.644 | 0.645 | 0 |
| 3.0 | 0.474, 0.504, 0.443, 0.492, 0.370, 0.563, **0.562\***, 0.513, 0.614, 0.477 | 0.460 | 0.483 | 0.501 | 0.498 | 0 |

(\*audited value, replacing the original cold-start-plateau artifact.)
Every point's audited mean and median remain well below the sweep's 0.70
stopping threshold, so the controller's "crossed, run exactly one more
point, then stop" decision (triggered at lambda3=2.0) is correct whether
evaluated on the naive or the audited statistic -- the audit does not
change the sweep's trajectory, only the honesty of what is reported for
it (and, for lambda3=3.0 specifically, the number used throughout Section
5's matched-mean-risk comparison: 0.501, not the naive 0.460). All Task 2
runs will be audited the same way as they land.

## Data and reproducibility

- `puffin/metrics/individual_fairness.py`: similarity definitions, raw
  IF-rate/dose-response, and the scale-free measure
  (`all_pairs_mean_abs_diff`, `scale_free_if`).
- `puffin/eval/plateau_audit.py`: the plateau-vs-genuine-collapse audit.
- `scripts/compute_individual_fairness.py` / `launch_individual_fairness.sh`:
  the per-dataset batch (raw IF-rate, unchanged).
- `scripts/generate_individual_fairness_report.py`: everything in this
  report, including the scale-free table and both matched-utility
  comparisons; regenerable from `reports/figures/individual_fairness/*.json`
  and the cached similarity matrices.
- `reports/figures/individual_fairness/scale_free_if_table.json`: full
  scale-free ratio + mean risk, every run, every similarity definition.
- Gowalla-SF-Bay's similarity-matrix cache is split into one file per
  definition (float32, ~20-27MB each) instead of one combined float64
  file (250MB, over GitHub's 100MB limit): `gowalla_sf_bay_similarities_
  {entropy,location_distribution,trajectory_shape,radius_of_gyration}.npz`.
- Every number traces to a `usable: true` protocol-tagged run; no stale
  or unaudited run contributes to this report.
