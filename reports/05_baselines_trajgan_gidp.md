# Checkpoint 5a — TrajGAN and GI-DP baselines (Phase 5, part 1)

**5-line summary:** Both third-party baselines the plan named are built,
tested, and validated on real GeoLife data. TrajGAN (LSTM-TrajGAN, Rao et
al. 2021) ran essentially out of the box after one trivial dependency pin
(`h5py==2.10.0`) -- the cleanest third-party outcome yet in this project --
so it was ported into `puffin/baselines/trajgan.py`. GI-DP turned out to
have no canonical open-source repo at all (not a blocker found after
cloning, just nothing to clone), so per the task's own explicit license to
use judgment, it's a clean in-house build of the classical Planar Laplace
mechanism (Andres et al. 2013) in `puffin/baselines/gi_dp.py`, verified
against the distribution's own closed-form mean (E[r]=2/eps, matched to
within 0.2%). Both produce real schema-valid synthetic/perturbed
trajectories on GeoLife; neither has been attacked with TUL/re-id/MIA yet
(that's a later phase's job per this task's own scope). Full account of
both: `CHANGES_FROM_ORIGINAL.md`.

## TrajGAN (LSTM-TrajGAN)

Confirmed repo identity first (GIScience 2021 paper, GitHub.com/GeoDS/
LSTM-TrajGAN) before touching anything. Cloned to `legacy/LSTM-TrajGAN/`
(not committed), isolated Python 3.6 venv with its own pinned deps. Its
own 20-epoch demo on its bundled Foursquare-NYC data ran with **zero code
changes** (D loss 32.3->8.2, G loss 2094->1986); the only real bug was in
`predict.py` (an h5py>=3.0 API incompatibility in Keras 2.2.4's weight
loader), fixed with one dependency pin. Because it ran, its generator/
discriminator/TrajLoss architecture was ported (not imported) into
`puffin/baselines/trajgan.py`: PyTorch, this project's unified schema,
`category` dropped (no venue-category field in this schema, same
reasoning as MainTUL's port), synthetic `cell_id` added via
re-tessellation (the GAN itself only outputs continuous lat/lon),
trajectories truncated to `max_traj_len=150` (GeoLife trajectories run up
to 92,645 points), and `scale_factor` computed via a robust 99.5th
percentile rather than the original's true max -- found necessary because
GeoLife's raw GPS data contains physically-impossible outlier fixes (lat
up to ~400 degrees) that would otherwise dominate the normalization.

Tested: `tests/test_trajgan.py` (3 checks, all pass), including an
end-to-end check that trained reconstruction error is less than half an
untrained generator's error on synthetic data.

Validated on real GeoLife (24,876,961 rows, 181 users; CPU, 30 epochs,
19m14s wall time): 14,788 train / 3,941 test trajectories, 370,179
synthetic points generated. G loss fell 24,875 -> 678 (~37x); D loss
fluctuated in a low range (final 0.0022) rather than monotonically
falling -- an expected unregularized-GAN dynamic given the original's own
`p_latlon=10` loss weighting, not broken training. Mean absolute lat/lon
reconstruction error on held-out trajectories fell 148.6/144.8 -> 9.7/16.5
degrees between epoch 2 and epoch 30 (15x/8.8x improvement); weekday/hour
argmax accuracy 0.185/0.090 vs. chance 0.143/0.042 -- real but modest.

## GI-DP (Geo-Indistinguishability / Planar Laplace)

No repo literally named "GI-DP" exists; GeoPrivacy (PyPI) only covers
single independent location reports, and privkit implements a different,
more elaborate adaptive variant -- neither is "the classical mechanism"
the task asked for. Built in-house directly from Andres et al. (2013)'s
own published formula: `puffin/baselines/gi_dp.py` samples Planar Laplace
noise via inverse-CDF (Lambert W function, lower branch), converts meters
to lat/lon degrees, and applies it independently per point (a documented,
deliberate scope limitation: the base mechanism has no trajectory-level
composition guarantee -- that's what temporal-correlation-aware
extensions like privkit's adaptive variant exist for, out of scope here).

Tested: `tests/test_gi_dp.py` (4 checks, all pass) -- empirical mean
radius matches the distribution's own theoretical `E[r]=2/epsilon` to
within 0.2%, noise is directionally unbiased, the epsilon privacy knob is
monotonic, and the output is a schema-valid dataframe with weekday/hour
untouched.

Validated on real GeoLife (24,876,961 rows, 181 users; ~105-112s/run, no
GPU needed), eps in {0.001, 0.01, 0.1}: empirical mean perturbation
distance matched the mechanism's own theoretical `E[r]=2/epsilon` almost
exactly (1999.7m/200.0m/20.0m vs. theoretical 2000/200/20m) -- confirms
correctness on real data, not just the test suite's toy data. Re-
tessellated `cell_id` vocabulary grows with noise, as expected: 349,900
distinct cells at eps=0.001 vs. the real tessellation's 85,074, down to
86,109 (close to the real count) at eps=0.1's much smaller ~20m average
perturbation.

## What's NOT done (explicitly out of this task's scope)

Neither mechanism has been attacked with TUL (`tul_maintul.py`), re-id
(`reid.py`), or MIA yet -- per the task's own instructions, that
evaluation is a later phase. Both now produce real, schema-valid output
on real GeoLife data, so that evaluation has real targets to run against
for the first time in this project (Phase 3 task 4's TUL validation had
to use GeoLife's own real trajectories as a stand-in, since neither
baseline existed yet).
