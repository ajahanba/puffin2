# Checkpoint 2c — Robustness checks on the 2b result, PI-requested

**Status: proceeding to Phase 3.** The core 2b finding (plain Mo-PAE does not
equalize risk) survives; one specific claim (GeoLife's two lowest deciles
getting *actively worse*) does not survive rigor and is retracted below —
correctly caught by exactly the checks the PI asked for before trusting it.

Code: `scripts/run_phase2c_robustness.py`, extensions to
`puffin/attacks/{windowing,reid,analysis}.py`. Data:
`reports/figures/phase2c/{geolife,foursquare,gowalla_nyc}/`.

## 1. GeoLife's "active harm" claim does not survive seed-pairing + bootstrap CIs

2b compared two independently-averaged 5-seed means (raw attacker's own 5
seeds vs. the frozen-encoder attacker's own, different, 5 seeds) and read
off a point estimate. 2c instead pairs each seed's raw result with *that
same seed's* post-mechanism result before averaging, then bootstraps 95%
CIs over users:

| decile | mean abs. reduction | 95% CI | excludes zero? |
|---|---|---|---|
| 0 (lowest entropy) | -0.047 | [-0.163, 0.087] | **no** |
| 1 | +0.022 | [-0.039, 0.083] | no |
| 2 | +0.014 | [-0.045, 0.075] | no |
| 3 | +0.022 | [0.002, 0.046] | **yes** |
| 4 | +0.021 | [0.003, 0.042] | **yes** |
| 5 | +0.110 | [0.037, 0.178] | **yes** |
| 6 | +0.029 | [0.002, 0.058] | **yes** |
| 7 | +0.009 | [-0.034, 0.045] | no |
| 8 | +0.017 | [-0.013, 0.040] | no |
| 9 (highest entropy) | +0.021 | [-0.008, 0.052] | no |

**Retraction:** 2b's "the two lowest-entropy deciles get worse" does not
hold up — decile 0's CI spans zero (not distinguishable from no effect),
and decile 1 flips to a slightly positive point estimate. Only the middle
deciles (3-6) show a statistically significant *improvement*. **The
revised, more defensible claim: plain Mo-PAE gives robust protection to
middle-entropy users, and no statistically detectable protection — neither
harm nor benefit — to the lowest- or highest-entropy users.** That's still
"under-protects the people who need it most" (decile 0 has nothing to show
for it), just not in the dramatic "makes them worse" form 2b reported.

This is exactly the failure mode bootstrapped CIs exist to catch — a noisy
point estimate from under-powered, seed-mismatched comparisons read as a
real effect. Foursquare and Gowalla-NYC's versions of this same check (below)
hold up fine, for contrast.

## 2. Regression with cluster-robust SEs (user-seed level, 5 obs/user)

| dataset | S_unc coef | 95% CI | n (obs / clusters) |
|---|---|---|---|
| GeoLife | -0.044 | [-0.065, -0.023] | 900 / 180 |
| Foursquare | -0.269 | [-0.302, -0.236] | 5415 / 1083 |
| Gowalla-NYC | -0.241 | [-0.294, -0.188] | 2935 / 587 |

Entropy's CI excludes zero on all three even after accounting for each
user's 5 attacker-seed observations not being independent (the classical
OLS SEs in 2b would have overstated precision here). This conclusion is
unchanged from 2b — just now on firmer statistical ground.

## 3. Attacker gap: chronological early stopping, and contemporaneous vs. future

Early stopping now uses each user's own chronologically-latest 10% of
*train* windows as validation (not a random subsample) — a more honest test
of "did this model learn something that generalizes forward in time."

**Contemporaneous (no time gap) vs. future (Phase 1's chronological split),
mean raw top-1 risk:**

| dataset | contemporaneous | future | ratio |
|---|---|---|---|
| GeoLife | 48.5% | 14.5% | 3.3x |
| Foursquare | 96.8% | 49.9% | 1.9x |
| Gowalla-NYC | 92.2% | 23.6% | 3.9x |

**The entropy gap holds under the contemporaneous split too, on all three**
— if anything more cleanly on GeoLife (decile 0 = 81.7% vs. decile 9 = 21.0%,
non-overlapping CIs) than under the future split. Foursquare/Gowalla-NYC's
contemporaneous deciles are compressed near a ceiling (mid-90s to high-90s%)
simply because contemporaneous check-in re-id is close to trivial for
everyone, so the *proportional* gap is harder to see there than on GeoLife,
but the ordering (low entropy = higher risk) still holds throughout.

**Does drift itself vary by decile? Yes, robustly (every single decile's CI
excludes zero on all three datasets) — but the DIRECTION is dataset-type
dependent, a genuinely new finding:**

| decile | GeoLife drift | Foursquare drift | Gowalla-NYC drift |
|---|---|---|---|
| 0 (lowest entropy) | **-0.513** (biggest drop) | -0.221 (smallest drop) | -0.438 (smallest drop) |
| 9 (highest entropy) | -0.157 (smallest drop) | **-0.666** (biggest drop) | **-0.769** (biggest drop) |

**GeoLife: low-entropy (regular) users drift the MOST over time** — their
short-term predictability decays fastest. **Foursquare and Gowalla-NYC: the
opposite** — low-entropy users drift the LEAST; their favorite-venue
patterns stay identifying far longer, while high-entropy check-in users'
identifiability erodes fastest. A plausible read: a GPS "routine" (a
commute, a season) is itself time-bound and changes with life circumstances,
while a check-in "regular spot" (a favorite coffee shop, gym) is a much more
durable habit. Not claiming this is the explanation — just that the pattern
is real, consistent within each dataset's 5 seeds, and opposite between GPS
and check-in data, which is itself worth keeping in mind for any Phase 3+
claim about "how strong an attacker can get with X months of stale data."

## Net effect on the paper's framing

Unchanged at the top level: plain Mo-PAE does not equalize risk, and
Checkpoint 2c did not surface a result where it *does* protect low-entropy
users adequately, so the PI's stop condition is not triggered. What changed:
GeoLife's specific "active harm" language is retracted in favor of a more
defensible "no significant protection" claim, and a new, reportable
finding — entropy-dependent, dataset-type-dependent temporal drift — is
now on record for whichever later phase wants it (Phase 8's scaling/data
recency discussion seems the natural home).

## Figures and data

`reports/figures/phase2c/{geolife,foursquare,gowalla_nyc}/summary.json`,
`per_user_future.csv`, `per_user_contemporaneous.csv`.
