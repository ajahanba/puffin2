# Checkpoint 04e: reference curve, recalibrated metrics, epoch-count fix, GeoLife 10-seed recheck

**5-line summary.** The GeoLife "ed\_soft dominates plain" claim from 04d
does **not** survive a clean, epoch-matched, 10-seed recheck: ed\_soft
shows a real, significant decile-0 risk reduction at λ4∈{0.3,1.0} with
**no** significant utility gain (not dominance), and a real utility
**cost** at λ4=3.0 (a straightforward trade-off). The plain-Mo-PAE
reference curve (λ3 swept, 40 epochs) is nearly flat on GeoLife, cleanly
monotonic on Foursquare, and noisier on Gowalla-NYC; comparing 04d's
still-20-epoch ed\_soft/floor points against this 40-epoch curve is only
**directional**, not conclusive, because those formulations were not
retrained at 40 epochs. The recalibrated leveling-down check finds 0 of
40 testable comparisons exceed a seed-to-seed null — likely underpowered
at 3 seeds, not evidence of absence. Foursquare's ed\_soft framing is
confirmed as "moving toward the no-privacy point," not leveling down. A
bigger ask — retraining all of 04d at 40 epochs — is flagged for sign-off,
not done here.

## 1. The epoch-count problem, and what is/isn't fixed

`scripts/diagnose_no_privacy_ceiling.py` found that `no_privacy`, `plain`,
and `ed_soft` are all still rising substantially at epoch 20 (GeoLife, one
seed), with their relative utility ORDER changing between epoch 20 and 40
(`ed_soft > no_privacy > plain` at 20; roughly tied at 40). There is no
early stopping anywhere in `run_phase4d_approved_batch.py` — every
mechanism uses a fixed epoch count. Full writeup: `CHANGES_FROM_ORIGINAL.md`.

**Fixed this round** (all at 40 epochs): the λ3 reference curve (new),
`no_privacy` refit (all 3 datasets), and GeoLife's 10-encoder-seed rerun of
`plain` + `ed_soft`.

**NOT fixed**: `ed_soft` on Foursquare/Gowalla-NYC and `floor` on GeoLife
remain at the original 20 epochs (04d's own runs, not retrained this
round). Every comparison below that involves one of these still-20-epoch
points against the new 40-epoch curve is flagged as **directional only**.
Retraining the rest of 04d at 40 epochs is a significant compute decision
(roughly doubling that batch's cost) — flagged here for the PI's sign-off,
not taken unilaterally.

## 2. The plain-Mo-PAE Pareto reference curve (item 1)

λ3 (`lambda_priv`) swept across {0.02, 0.1, 0.2, 0.5, 0.8} at 40 epochs, 3
encoder seeds; λ3=0.3 reuses 04d's existing 20-epoch point for display but
is **excluded from interpolation** below (see §3 for why). Full data:
`reports/figures/phase4e_analysis/reference_curve_comparisons.json`.

**GeoLife's curve is nearly flat.** Utility ranges only 0.722–0.758 and
decile-0 risk only 0.337–0.348 across the entire λ3 sweep (0.02 to 0.8) —
once properly converged, λ3 barely moves either utility or risk on this
dataset over this range. This was not anticipated going in; it means a
"Pareto frontier" comparison on GeoLife is comparing against an
essentially flat reference, not a sloped trade-off curve.

**Foursquare's curve is clean and monotonic**: utility falls smoothly from
0.210 (λ3=0.02) to 0.051 (λ3=0.8) as the privacy weight rises, and risk
falls in step (0.685→0.149). The reused 20-epoch λ3=0.3 point (utility
0.111) is a visible outlier against this trend — it would need to be
≈0.15–0.17 to fit smoothly between its neighbors, confirming it is
undertrained relative to the rest of its own curve, not a real kink in the
frontier.

**Gowalla-NYC's curve is noisier**, consistent with its much smaller user
count (272 after filtering): utility clusters around 0.24–0.27 for
λ3∈{0.1,0.2,0.3,0.5} with no clean monotonic trend, then drops sharply at
λ3=0.8 (0.173).

**The no-privacy ceiling, corrected**:

| dataset | ceiling @ 20ep (stale) | ceiling @ 40ep (corrected) |
|---|---|---|
| GeoLife | 0.623 ± 0.014 | 0.742 ± 0.005 |
| Foursquare | 0.280 ± 0.002 | 0.214 ± 0.002 |
| Gowalla-NYC | 0.322 ± 0.005 | 0.251 ± 0.001 |

GeoLife's ceiling rose with more training (as expected for an undertrained
point). Foursquare and Gowalla-NYC's ceilings **fell** — at first glance
surprising, but both corrected values fit smoothly into their own dataset's
λ3 curve (Foursquare: 0.214 slots in just above λ3=0.02's 0.210, continuing
the monotonic trend to its zero-privacy-pressure limit; Gowalla-NYC: 0.251
sits between λ3=0.02's 0.250 and λ3=0.1's 0.241). The stale 20-epoch
values were inflated by `no_privacy`'s faster early convergence (fewer
competing loss terms), the same confound diagnosed in §1 — not by a
retraining regression. **Caveat**: the 40-epoch choice was verified by a
per-epoch curve only on GeoLife; Foursquare and Gowalla-NYC's 40-epoch
values are not independently confirmed to be converged (vs. still moving,
in either direction) — a follow-up per-epoch diagnostic on those two
datasets would close this gap but was not run this round.

## 3. Comparing 04d's formulations against the curve (directional only)

Every comparison below interpolates the formulation's utility against the
**40-epoch-only** points of its dataset's curve (the stale 20-epoch λ3=0.3
point is excluded from interpolation, per §2's finding that it's an
outlier). Full data in the same JSON as §2.

**GeoLife: every comparison is `unresolvable`.** All six 04d points
(ed\_soft λ4∈{0.3,1.0,3.0}, floor λ4∈{0.3,1.0,3.0}, utility 0.516–0.654)
fall entirely below the 40-epoch curve's minimum utility (0.722) — they
cannot be honestly placed on this curve at all. This is itself informative:
04d's GeoLife points are not merely "a bit lower," they are in a different
utility regime than a properly-converged plain Mo-PAE, reinforcing that
GeoLife's own `ed_soft`/`floor` points need the same 40-epoch retraining
`plain`/`no_privacy` got this round before any curve-based claim can be made
for them. (The dominance question for GeoLife `ed_soft` specifically is
answered cleanly in §4 instead, via the matched-epoch 10-seed rerun.)

**Foursquare**: `ed_soft` λ4=0.3 (utility 0.134, within the curve's range)
is `worse_than_curve` — its risk (0.554) is higher than the curve predicts
at that utility (0.504). λ4∈{1.0,3.0} (utility 0.221, 0.233) are
`unresolvable` — both **exceed even the corrected 40-epoch no-privacy
ceiling (0.214)**, the same anomaly §1 diagnosed, now shown to be
unresolved for Foursquare's `ed_soft` specifically (it was not retrained
this round).

**Gowalla-NYC**: `ed_soft` λ4=0.3 is `worse_than_curve` (risk 0.557 vs.
curve's 0.534); λ4=1.0 is `beats_curve` (risk 0.548 vs. curve's 0.636);
λ4=3.0 is `unresolvable` (utility 0.169 below the curve's minimum, 0.173).

All of the above are flagged `epoch_mismatch_caveat` in the JSON:
directional signal only, not a conclusion, until `ed_soft` (Foursquare,
Gowalla-NYC) and `floor` (GeoLife) are retrained at 40 epochs.

## 4. GeoLife 10-encoder-seed recheck (item 6) — the dominance claim is retracted

Both `plain` and `ed_soft` (λ4∈{0.3,1.0,3.0}) retrained at 40 epochs, 10
encoder seeds, on GeoLife. Per-seed utility and per-seed decile-0 risk
(10 values each) compared via `paired_bootstrap_test`:

| λ4 | utility: ed\_soft − plain | CI excludes 0? | decile-0 risk: plain − ed\_soft | CI excludes 0? | dominates? |
|---|---|---|---|---|---|
| 0.3 | +0.007 | **no** | +0.113 | **yes** | **No** |
| 1.0 | +0.004 | **no** | +0.138 | **yes** | **No** |
| 3.0 | −0.020 | **yes** (worse) | +0.149 | **yes** | **No** |

`plain` (10 seeds, 40 epochs): utility 0.731 ± 0.019, decile-0 risk 0.359.

**04d's 3-seed "dominates" claim does not survive.** At λ4∈{0.3,1.0},
`ed_soft` gives a real, statistically significant reduction in decile-0 risk
with **no** detectable utility cost (the utility difference is not
distinguishable from zero at 10 seeds) — a genuine, partial privacy benefit,
but not dominance (dominance needs utility to be significantly *higher*,
not merely not-significantly-lower). At λ4=3.0, `ed_soft` trades a
significant utility cost for the risk reduction — an ordinary trade-off,
the opposite of dominance. The original claim was built on 3 seeds at an
unconverged epoch count; both of those turn out to have mattered.

## 5. Recalibrated leveling-down check (item 3), rerun

`puffin/metrics/leveling_down.py`'s absolute-risk/seed-to-seed-null version
was rerun on 04d's existing (20-epoch) data in the previous commit
(`011e202`): **0 of 40 computed comparisons** show `harmed_rate_exceeds_null`
— every one of the earlier "5–6/10 deciles" (GeoLife floor) and "10/10
deciles" (Gowalla-NYC ed\_soft λ4=0.3) leveling-down findings fails to
clear even the lowest bar (a harmed rate distinguishable from the null)
once tested against a real noise floor. This is most likely an
**underpowered test** — only 3 encoder seeds give a thin null (6 ordered
pairs per user) — not proof that no such pattern exists; §4's much better-
powered 10-seed paired test on GeoLife's `ed_soft` vs. `plain` (decile-0
only, not per-decile) is the stronger evidence available this round, and it
shows a real aggregate effect (§4) even though the per-user leveling-down
classifier (this section) could not confirm or rule out redistribution
within that effect. A full per-decile leveling-down check using GeoLife's
rich 10-seed null (90 ordered pairs) was not run this round — a natural,
low-cost follow-up.

## 6. Foursquare ed\_soft restatement (item 5), confirmed

Per the PI's correction (already applied in commit `011e202`): raising
`ed_soft`'s λ4 on Foursquare moves the operating point toward the
no-privacy reference (utility and risk both rise together, e.g. utility
0.111→0.233, decile-0 risk 0.487→0.686) — an ordinary privacy-utility
trade-off, not a leveling-down pathology. §3's finding that Foursquare's
`ed_soft` can exceed even the *corrected* no-privacy ceiling sharpens this:
`ed_soft` at high λ4 is not just trading toward the no-privacy point, in
utility terms it can be mistakenly read as exceeding it, for the same
epoch-count reason diagnosed in §1.

## Figures and data

`reports/figures/phase4e_refcurve/` (λ3 sweep, 40ep), `phase4e_refit40/`
(no\_privacy refits, 40ep), `phase4e_10seed/` (GeoLife plain+ed\_soft,
40ep, 10 seeds), `phase4e_analysis/{reference_curve_comparisons,
disparity_ratio_and_difference,leveling_down_recalibrated}.json`. Scripts:
`scripts/diagnose_no_privacy_ceiling.py`, `scripts/launch_phase4e_batch.sh`,
`scripts/analyze_04e_reference_curve.py`,
`scripts/recompute_04e_leveling_and_difference.py` (all pure analysis
except the launcher, which is the only new training this round).

**Stopping and reporting here, per the PI's instruction.** Attacker-in-the-
loop (queued, built and smoke-tested in
`puffin/fairness/attacker_in_the_loop.py`) is launched at lower priority
immediately after this report, evaluated against this checkpoint's
reference curve once it completes — not analyzed as part of this report.
