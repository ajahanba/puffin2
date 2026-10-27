# Checkpoint 04d: approved batch -- results, decile-reduction metrics,
# in-model-vs-external mechanistic comparison, rebuilt 04c fronts, leveling-down check

All 21 PI-approved runs finished (plain Mo-PAE, the no-privacy
utility-ceiling reference, ed_soft[0, 0.3, 1.0, 3.0], and GeoLife's
fixed floor_loss[0.3, 1.0, 3.0] -- GeoLife, Foursquare, Gowalla-NYC, each
>=3 encoder-training seeds x 5 attacker seeds, hardened external
attacker, strongest-per-user). Analysis script: pure re-analysis of data
already on disk, no retraining -- `scripts/analyze_04d_approved_batch.py`,
output in `reports/figures/phase4d_analysis/`.

**Headline finding: the no-privacy reference shows the privacy mechanism
itself widens the decile0/decile9 disparity on two of three datasets,
rather than narrowing it, and raising `ed_soft`'s fairness weight on
Foursquare makes the worst-off decile's actual post-mechanism risk
*increase*, even as it raises utility.** Both are reported in detail
below. Per the PI's instruction, this is a stop-and-report checkpoint --
the attacker-in-the-loop grid (queued, see `PROGRESS.md`) is not
launched by this report.

## 1. Decile-reduction metrics, every run

Headline statistics only (decile 0 = lowest entropy / most identifiable,
decile 9 = highest entropy); full per-decile tables in
`reports/figures/phase4d_analysis/decile_reduction_metrics.json`.
RRD = residual-risk-disparity ratio (post-mechanism risk, decile0 /
decile9), bootstrap 95% CI via independent per-decile resampling.

### GeoLife (n=178 after the >=10-window filter)

| run | utility | rel. reduction, decile 0 | rel. reduction, decile 9 | RRD |
|---|---|---|---|---|
| plain | 0.532 ± 0.052 | +0.014 [-0.133, +0.176] | +0.371 [-0.042, +0.625] | 8.08 [3.72, 14.11] |
| no_privacy | 0.623 ± 0.014 | -0.032 [-0.173, +0.104] | +0.340 [-0.045, +0.604] | 8.02 [3.79, 13.93] |
| ed_soft λ4=0.3 | 0.641 ± 0.039 | +0.160 [+0.016, +0.295] | +0.239 [-0.076, +0.481] | 5.57 [2.67, 9.40] |
| ed_soft λ4=1.0 | 0.654 ± 0.039 | +0.255 [+0.079, +0.408] | +0.245 [-0.151, +0.516] | 5.00 [2.49, 8.38] |
| ed_soft λ4=3.0 | 0.631 ± 0.014 | +0.302 [+0.196, +0.419] | +0.299 [-0.019, +0.534] | 5.08 [2.44, 8.76] |
| floor λ4=0.3 | 0.551 ± 0.047 | -0.031 [-0.158, +0.096] | +0.347 [-0.097, +0.614] | 8.19 [3.95, 14.20] |
| floor λ4=1.0 | 0.545 ± 0.055 | -0.010 [-0.154, +0.114] | +0.425 [+0.142, +0.603] | 8.82 [4.56, 14.63] |
| floor λ4=3.0 | 0.516 ± 0.085 | -0.008 [-0.128, +0.111] | +0.473 [+0.210, +0.650] | 9.68 [4.74, 16.43] |

Plain Mo-PAE gives decile 0 **no statistically distinguishable
protection at all** (CI straddles zero). ed_soft is the only
formulation that gives decile 0 a clearly nonzero reduction (CI
excludes zero at λ4=1.0, 3.0) and the only one that lowers RRD (8.08 →
~5.0-5.6). The fixed floor_loss does the opposite of its intended
purpose: decile 0 stays statistically unchanged at every λ4, decile 9
(already the best-protected decile) gets *more* protection as λ4 rises,
and RRD gets **worse**, not better (8.19 → 9.68). The floor_loss fix
(Section 2's gradient issue) was real and necessary, but the term still
pushes down whichever user's in-model confidence crosses the threshold
first -- which this batch shows is not preferentially decile 0's users.

### Foursquare (n=1083)

| run | utility | rel. reduction, decile 0 | rel. reduction, decile 9 | RRD |
|---|---|---|---|---|
| plain | 0.111 ± 0.005 | +0.384 [+0.331, +0.439] | +0.897 [+0.866, +0.925] | 16.02 [10.84, 23.71] |
| no_privacy | 0.280 ± 0.002 | +0.096 [+0.065, +0.127] | +0.310 [+0.250, +0.374] | 3.43 [2.74, 4.32] |
| ed_soft λ4=0.3 | 0.134 ± 0.021 | +0.300 [+0.243, +0.358] | +0.850 [+0.810, +0.885] | 12.38 [8.61, 17.59] |
| ed_soft λ4=1.0 | 0.221 ± 0.011 | +0.135 [+0.095, +0.179] | +0.600 [+0.543, +0.658] | 5.68 [4.37, 7.38] |
| ed_soft λ4=3.0 | 0.233 ± 0.004 | +0.133 [+0.096, +0.171] | +0.568 [+0.508, +0.628] | 5.28 [4.10, 6.85] |

The no-privacy reference's own RRD (3.43) is the baseline disparity
already present in an *undefended* learned representation. Plain
Mo-PAE's RRD (16.02) is **4.7x larger than that undefended baseline** --
the privacy objective itself manufactures most of this dataset's
disparity, it does not inherit it. Raising ed_soft's λ4 does lower RRD
(16.02 → 5.28), but not by protecting decile 0 better: decile 0's
*absolute* post-mechanism risk rises monotonically with λ4 (0.487 →
0.554 → 0.684 → 0.686, Section 3's Pareto table), and its relative
reduction shrinks correspondingly (+0.384 → +0.133). The RRD drop is
driven entirely by decile 9 losing protection faster than decile 0
(+0.897 → +0.568) while *utility rises* (0.111 → 0.233). This is a
leveling-down pattern in the strict sense found in the formal check
(Section 4): the gap narrows by making the best-off decile worse off,
while the worst-off decile's real risk gets measurably worse, not
better, and the run's headline utility number improves throughout --
exactly the configuration where an aggregate utility/fairness-loss
readout would look like unambiguous progress.

### Gowalla-NYC (n=272 after the >=10-window filter)

| run | utility | rel. reduction, decile 0 | rel. reduction, decile 9 | RRD |
|---|---|---|---|---|
| plain | 0.269 ± 0.035 | +0.240 [+0.115, +0.371] | +0.634 [+0.519, +0.731] | 7.26 [4.24, 11.71] |
| no_privacy | 0.322 ± 0.005 | +0.200 [+0.077, +0.343] | +0.334 [+0.245, +0.430] | 4.12 [2.63, 6.27] |
| ed_soft λ4=0.3 | 0.263 ± 0.005 | +0.253 [+0.149, +0.360] | +0.555 [+0.452, +0.650] | 5.82 [3.61, 9.08] |
| ed_soft λ4=1.0 | 0.239 ± 0.026 | +0.266 [+0.160, +0.381] | +0.596 [+0.527, +0.664] | 6.29 [3.86, 10.02] |
| ed_soft λ4=3.0 | 0.169 ± 0.043 | +0.244 [+0.154, +0.345] | +0.629 [+0.540, +0.719] | 7.13 [4.24, 11.69] |

Same direction as Foursquare (no-privacy's RRD of 4.12 is roughly half
plain's 7.26 -- the mechanism again widens, not narrows, the baseline
disparity) but a much weaker, non-monotonic λ4 trend: RRD dips at
λ4=0.3 (5.82) then rises back toward plain's level by λ4=3.0 (7.13),
and decile 0's relative reduction is flat within noise across all four
λ4 values (0.240-0.266, all CIs overlapping heavily). Utility falls
monotonically (0.269 → 0.169) for no corresponding privacy gain at
λ4=3.0 -- a straightforwardly bad trade at the high end, not even a
leveling-down pattern, just a worse point on every axis than plain.

**Cross-dataset pattern**: decile 0 is the worst-off (highest
post-mechanism risk) decile in literally every one of the 21 runs,
with no exception -- `worst_decile=0` in every entry of
`pareto_fronts_rebuilt.json`. The no-privacy-vs-plain RRD comparison
(mechanism amplifies baseline disparity on Foursquare and Gowalla-NYC,
flat on GeoLife: 8.02 → 8.08) is itself dataset-dependent and not
currently explained -- flagged as an open question, not a settled claim.

## 2. In-model adversary vs. external attacker, by decile (mechanistic finding)

Per the PI's request: the in-model adversary's own confidence
(`bundle.id_recognizer`, evaluated directly, same per-sample confidence
calculation used during training) next to the external attacker's
top-1 on the same encoder features, by decile, for the `plain`
mechanism (the case floor_loss's inertness was diagnosed on).
Full table: `reports/figures/phase4d_analysis/inmodel_vs_external.json`.

| dataset | chance level | decile 0: external / in-model | decile 9: external / in-model |
|---|---|---|---|
| GeoLife | 0.0055 | 0.345 / 0.0246 | 0.045 / 0.0113 |
| Foursquare | 0.00092 | 0.487 / 0.0521 | 0.031 / 0.0020 |
| Gowalla-NYC | 0.000139 | 0.566 / 0.0776 | 0.082 / 0.0063 |

In every dataset and every decile, the in-model adversary's confidence
sits far below the external attacker's accuracy on the identical
features, and its own decile-to-decile range is far smaller than the
external attacker's. GeoLife is the cleanest case: in-model confidence
stays within a narrow 0.0085-0.0246 band (1.5x-4.5x chance) across all
10 deciles while the external attacker ranges 0.045-0.345 (8x-63x
chance) -- this is exactly why floor_loss's fixed threshold (originally
0.3, now chance-relative) could never engage: the in-model adversary
never gets confident enough, on *any* decile, for a confidence-based
fairness term to tell deciles apart. Foursquare and Gowalla-NYC are
less clean: both show an elevated in-model confidence specifically at
decile 0 (0.052 and 0.078 respectively) that the other 9 deciles don't
share -- the in-model adversary isn't perfectly flat on those two
datasets, it does leak a little decile-0-specific signal, just 7-9x
weaker than the external attacker's at the same decile, and flat
elsewhere. **A loss-based fairness term built on the in-model adversary
is targeting the wrong risk surface on all three datasets, to varying
degrees** -- this is the mechanistic version of the empirical finding
in Section 1 (floor_loss doesn't move decile 0's real risk because the
confidence it watches barely distinguishes decile 0 from any other
decile).

## 3. Rebuilt Pareto fronts (plain Mo-PAE and no-privacy overlaid)

Full points: `reports/figures/phase4d_analysis/pareto_fronts_rebuilt.json`.
Every run's worst decile is decile 0, so "decile-0 risk" and "max-min
risk" are identical here (unlike 04c's single-encoder-seed grid, no
held-out-seed split was needed to pick the worst decile -- entropy
decile is fixed by the data, not by attacker/encoder randomness).

`ed_soft` now has a real utility axis (3 encoder seeds, properly
logged) in place of 04c's unusable version, and plain Mo-PAE / the
no-privacy ceiling are now both plottable (closing 04c's blocking gap).
In point estimates, GeoLife's ed_soft dominates plain at all three λ4
values (higher utility *and* lower decile-0 risk: e.g. λ4=1.0, utility
0.654 vs. 0.532, risk 0.258 vs. 0.345); this is the one case in the
batch where "dominates" survives the PI's bar (utility CIs for ed_soft
λ4≥1.0 and plain do not overlap: 0.654±0.039 vs. 0.532±0.052, and
decile-0's relative-reduction CI excludes zero only for ed_soft). On
Foursquare and Gowalla-NYC, ed_soft never dominates plain in either
metric at the same time (Section 1) -- every Foursquare/Gowalla ed_soft
point is a trade-off, several of them (Section 1, Gowalla λ4=3.0;
Foursquare's decile-0 risk increase) a trade *against* the party the
mechanism exists to protect.

The 5 single-encoder-seed Phase 4 formulations (lipschitz, variance,
group_dro, per_user_minimax, group_adversarial) are included in the
rebuilt fronts with plain/no-privacy now plottable alongside them for
the first time, but every point carries an explicit
`n_encoder_seeds=1` vs. plain's 3 -- a seed-count mismatch, not a
like-for-like comparison, and not resolved by this round (out of the
PI's approved retraining scope).

## 4. Leveling-down check

Per the PI's instruction, run against plain Mo-PAE at matched utility
(±0.02), by decile, for every (mechanism, λ4) point where a match
exists. Full results: `reports/figures/phase4d_analysis/leveling_down_rebuilt.json`.

**Only 40 of 78 attempted (mechanism, λ4, decile) comparisons could be
computed; 38 were skipped because utility wasn't matched within ±0.02** --
ed_soft's utility moves too much with λ4 in this batch (Section 1) to
stay within the tolerance except for one point (Gowalla-NYC, λ4=0.3).
The 40 computed comparisons:

- **GeoLife floor** (all 3 λ4, matched within tolerance): 5-6 of 10
  deciles per λ4 flagged `leveling_down=True` (harmed users were
  disproportionately the ones plain already protected well), mean
  fraction of users harmed per decile 0.39-0.51. Consistent with
  Section 1's finding that floor_loss helps decile 9, not decile 0.
- **Gowalla-NYC ed_soft λ4=0.3** (the only ed_soft point that matched):
  **10 of 10 deciles** flagged `leveling_down=True`, mean fraction
  harmed 0.494 -- a clean, consistent leveling-down signal at the one
  point where the comparison could actually be run.

No ed_soft point on GeoLife or Foursquare could be checked this way at
all (utility moved too much). This is itself a limitation worth
flagging: the ±0.02 utility-matching protocol, inherited from 04c,
turns out to be too strict for this batch's own data, where λ4 reliably
shifts utility by far more than 0.02 in most (dataset, mechanism)
pairs. Every comparison that *could* be run pointed toward leveling
down, not toward lifting the worst-off decile; none pointed the other
way.

## 5. Encoder-training seeds

Every 04d run (plain, no_privacy, ed_soft, floor) used 3 independent
encoder-training seeds (0, 1, 2), reported as `utility_per_encoder_seed`
in each run's `summary.json` and as `utility_std` above. The 5 original
Phase 4 formulations (lipschitz, variance, group_dro, per_user_minimax,
group_adversarial) remain single-encoder-seed, per the PI's approved
scope for this round (retraining them was not part of the approved
batch).

## 6. Known gaps in this batch

- 5 of the 21 runs (GeoLife: ed_soft λ4∈{0,0.3,1.0,3.0}, no_privacy)
  were launched before the in-model-confidence capture feature was
  added to the driver and lack `inmodel_confidence` in their per-user
  CSVs -- Section 2's mechanistic comparison uses the 3 `plain` runs
  (all 3 datasets have it) and does not need these 5, but they can't be
  backfilled into Section 2's table without a rerun if that becomes
  load-bearing later.
- The no-privacy-vs-plain RRD-amplification pattern (Section 1) is
  dataset-dependent (present on Foursquare/Gowalla-NYC, absent on
  GeoLife) with no mechanistic explanation yet -- flagged, not
  resolved.

## Figures and data

`reports/figures/phase4d/<dataset>_<mechanism>_lambda<l4>/{summary.json,per_user_04d.csv}`
(21 runs, raw outputs) and `reports/figures/phase4d_analysis/{decile_reduction_metrics,inmodel_vs_external,pareto_fronts_rebuilt,leveling_down_rebuilt}.json`
(this report's analysis). Script: `scripts/analyze_04d_approved_batch.py`
(pure analysis, no training).

**Stopping and reporting here, per the PI's instruction.** The
attacker-in-the-loop grid (72 configs, queued per the PI's follow-up
request, built and smoke-tested in `puffin/fairness/attacker_in_the_loop.py`)
is not launched by this report -- see `PROGRESS.md` for its exact
scope and launch command.
