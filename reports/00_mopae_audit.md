# Checkpoint 0 — Mo-PAE audit

**Status: ⛔ awaiting PI review before Phase 1 starts.**

## What was compared

1. **Upstream** `mopae.py` (https://github.com/YutingZhan/Mo-PAE, cloned into
   `legacy/Mo-PAE_upstream/`), run unmodified except `EPOCHS=1000 → 3`, in an
   isolated Python 3.10 / TensorFlow 2.10 venv (`.venv-legacy-tf`), on CPU.
2. **Utkarsh's PUFFIN fork**, read directly from erebor
   (`/home/NETID/ud21/Mo-PAE_og_fairness/`, `Mo-PAE__fixed_naive/`) — not
   re-run here, just audited by reading `fairmopae.py`, `functions_mopae.py`,
   `models_mopae.py`, `data_preprocess.py`.
3. **This repo's PyTorch port** (`puffin/models/mopae_torch.py`,
   `puffin/data/geolife.py`), `legacy` mode (faithfully reproduces upstream)
   vs `fixed` mode (addresses issues #2–#4, #6, #7 below), 3 epochs, seed 0,
   CPU, on the same GeoLife sample (`data/raw/geolife_sample/`).

All three share one GeoLife sample (182 users' worth of grid-cell sequences,
`sample_trainingdata/GeoLifeBeijing_grid_{train,test}.csv`).

## Issue-by-issue: does PUFFIN inherit mopae.py's problems?

| # | Issue | Upstream `mopae.py` | Utkarsh's PUFFIN (`og_fairness` / `fixed_naive`) | This port |
|---|---|---|---|---|
| 1 | Reports accuracy on **train** set only | Yes — `get_accuracies(y_pre, y_train, 10)` | `og_fairness`: eval code is dead (inside a never-executed triple-quoted block). `fixed_naive`: added a real test-split eval loop, but it silently **drops any user with zero test windows** instead of reporting them | `legacy` mode reports train only, to match; `fixed` mode reports both, see below |
| 2 | Windows span users, labeled by the window's first point | Yes | Yes, unchanged in both forks | `legacy`: reproduced. `fixed`: windows built within one (user, trajectory) only |
| 3 | 80/20 split happens **after** windowing, so it doesn't align with per-user time | Yes | Yes, unchanged | `legacy`: reproduced. `fixed`: each user's own trajectory split 80/20 by time *before* windowing |
| 4 | Input is one LSTM timestep of `context` raw location-ID floats; weekday/hour unused | Yes | Yes, unchanged | `legacy`: reproduced. `fixed`: real `context`-length sequence, `nn.Embedding` for location + weekday + hour |
| 5 | Softmax output run through `categorical_crossentropy(..., from_logits=True)` — double softmax | Yes, confirmed: training log shows Keras's own warning, *"received `from_logits=True`, but the output... does not represent logits"* | **No** — both forks already use `from_logits=False`. This bug did not carry into PUFFIN | `legacy`: reproduced via `_double_softmax_xent`. `fixed`: single correct `cross_entropy` on logits |
| 6 | Reconstruction target is a location-ID magnitude, not a real distance | Yes — `minkowski_distance` on location ids (euc=53820, man=11034818 in the run below — meaningless units) | Dead code in both forks (never executed) | `legacy`: reproduced (meaningless by construction). `fixed`: decoder reconstructs (lat, lon) grid coordinates instead |
| 7 | λ signs baked into the constants; one shared Adam optimizer for all four sub-networks | Yes | Yes. One real fix found: `og_fairness` has `lambda4 = +LAMBDA_ALL[3]`, `fixed_naive` flips it to `lambda4 = -1*LAMBDA_ALL[3]` — Utkarsh caught a sign bug in the *fairness* term specifically. λ1/λ3 and the shared optimizer are untouched in both | `legacy`: reproduced. `fixed`: λ's are all positive, the adversarial minus sign is explicit in the encoder's loss expression, four independent optimizers |
| 8 | Privacy measured only by the in-model adversary (the same one the encoder was trained to fool) | Yes | Yes — `fixed_naive`'s per-user validation still scores against `id_recognizer`'s own output | Unchanged — out of scope for Phase 0, this is Phase 3's external-attacker suite. Don't read `internal_reid` below as a privacy guarantee |

**Net: every issue upstream is still live in both of Utkarsh's directories
except #5, and #1/#7 are each partially addressed.** Any number from the
rejected PUFFIN draft inherits #2, #3, #4, #6, #8, and the shared-optimizer
half of #7.

## Numbers: legacy vs fixed, train vs test (3 epochs, seed 0, CPU, GeoLife sample)

Upstream `mopae.py`, unmodified, 3 epochs (train-only, as the script only
ever evaluates on train — this is issue #1 in action):

| | top1 | top5 | top10 |
|---|---|---|---|
| utility (train) | 15.2% | 32.4% | 41.2% |
| internal re-id (train) | 10.8% | 36.1% | 50.0% |

This port, `legacy` mode (same windowing/split/input encoding as upstream, so
it should resemble the row above) vs `fixed` mode (issues #2–#4, #6, #7
addressed):

| | utility top1 | utility top5 | utility top10 | re-id top1 | re-id top5 | n windows |
|---|---|---|---|---|---|---|
| legacy — train | 7.0% | 21.2% | 32.3% | 0.01% | 0.14% | 95,721 |
| legacy — test | 7.0% | 20.3% | 30.3% | 0.0% | 1.4% | 23,930 |
| fixed — train | 15.4% | 36.4% | 47.1% | 11.7% | 39.4% | 77,873 |
| fixed — test | 15.7% | 37.0% | 48.2% | 11.4% | 40.4% | 10,815 |

(`legacy` and upstream aren't expected to match exactly — different framework,
init, optimizer internals — but `legacy`'s window counts, 95,721/23,930,
match upstream's `(95721, 1, 10) (23930, 1, 10)` log line exactly, confirming
the windowing bug is faithfully reproduced.)

### What this already shows, even at 3 epochs

- **Issue #5 has a real, measurable cost, independent of #2–#4.** `legacy`
  mode's internal re-id is at or *below* the 1/145≈0.7% random-guess rate —
  the double-softmax bug doesn't just miscompute a number, it cripples the
  gradient signal the id_recognizer needs to learn anything at all. Since
  Utkarsh's fork already avoids this bug, this isn't a problem for PUFFIN's
  existing numbers, but it is a warning for anyone re-deriving code from
  upstream `mopae.py` directly.
- **`fixed` mode's re-id accuracy (11–40%) is wildly different from both
  `legacy`'s (≈0%) and upstream's train-only 10.8%/36.1%.** None of these
  are comparable to each other — different architectures, different (and in
  `legacy`'s/upstream's case, leaky) data splits. The absolute numbers here
  are not a result, just a sanity check that `fixed` mode's adversary can
  learn a real signal.
- **Fixing the windowing/split leakage costs real data.** `fixed` mode has
  ~19% fewer training windows (77,873 vs 95,721) and ~55% fewer test windows
  (10,815 vs 23,930) than `legacy`'s, because per-(user, trajectory) windows
  can't be built across short trajectories' boundaries. Phase 1's "record how
  many users/windows each filter drops" should expect this kind of loss on
  every dataset, not just GeoLife.
- **No train/test gap is visible yet in `fixed` mode** (test is actually
  marginally higher than train) — expected noise at 3 epochs on one seed, not
  a finding. A real test/train comparison needs the full Phase 1 pipeline
  (≥5 seeds, real epoch budget).
- `fixed` mode is slower per epoch on CPU (262s vs 149s for `legacy`, 3
  epochs) — four separate optimizers and the embedding lookup add overhead.
  Worth tracking once Phase 8's scaling study starts.

## Decision this checkpoint needs from the PI

The old PUFFIN draft's numbers came from code that still has issues #2, #3,
#4, #6, #8 (and the shared-optimizer half of #7) exactly as described above.
Per the plan: **"If it was built on mopae.py, every PUFFIN number in the old
draft inherits these issues."** It was, and they do.

Suggested framing for the paper (for the PI to confirm or redirect): describe
the old PUFFIN result as a *pilot* that motivated this work, not as a
baseline number to beat — Phase 1 onward should produce the first trustworthy
numbers for this dataset/method pair.

Reused artifacts worth keeping from Utkarsh's fork: the fairness-loss
variants (`ed`, `matches`, `naive`, `confidence` in `models_mopae.py`) are
useful historical reference for what PUFFIN v1 tried, but don't match the
plan's Phase 4 formulation (`L_pair_threshold`, `L_lipschitz`, `L_floor`) and
aren't reused as-is.

## Environment notes for whoever runs Phase 1+ on erebor

- `/home` is at 99% capacity (6.8G free of 390G) — this repo and all data
  live under `/data/mashhadi/puffin2` instead (`/data` has 462G free).
- No conda/pyenv on this account; system Python 3.9–3.12 are all present as
  `/usr/bin/python3.X`. Two venvs: `.venv` (3.10, PyTorch 2.14, CPU-only —
  the installed driver, 535.54, is too old for the CUDA 13 wheel pip picked)
  and `.venv-legacy-tf` (3.10, TF 2.10, CPU-only, for `legacy/Mo-PAE_upstream`
  only). **Before any real training run, reinstall `torch` with a CUDA 12.x
  build matching driver 535 so the four A4500 GPUs are actually used** —
  everything in this checkpoint ran on CPU.
