# Changes from the original Mo-PAE / PUFFIN code

Source of truth for "original": https://github.com/YutingZhan/Mo-PAE `mopae.py`
(cloned into `legacy/Mo-PAE_upstream/`), cross-checked against Utkarsh Darbari's
PUFFIN fork (`Mo-PAE_og_fairness/`, `Mo-PAE__fixed_naive/`, both on erebor under
`/home/NETID/ud21/`).

## Phase 0 (this port, `puffin/models/mopae_torch.py`, `puffin/data/geolife.py`)

Both `legacy` mode and `fixed` mode live side by side so the two can be compared
on identical data. `legacy` mode intentionally reproduces every bug below; it
is not meant to be used for any real result, only as the audit baseline.

| # | Issue (see `reports/00_mopae_audit.md`) | legacy mode | fixed mode |
|---|---|---|---|
| 1 | Eval only on train | reports train metrics only used for comparison, but splits train/test the buggy way | evaluates on genuinely held-out per-user test windows |
| 2 | Windows span users, labeled by first point | `legacy_windows()`: slides over the full concatenated location array, target id = window's first point | `fixed_windows()`: windows built within a single (user, trajectory) group; never crosses a user or trajectory boundary |
| 3 | Split after windowing | split happens on the already-windowed arrays, by position | each user's own trajectory is split 80/20 by time *before* windowing |
| 4 | Non-sequential input, ids as raw floats | `LegacyEncoder`: one LSTM timestep, `context` raw floats as "features" | `FixedEncoder`: real `context`-length sequence, `nn.Embedding` for location + weekday + hour |
| 5 | `from_logits=True` on a softmax output (double softmax) | reproduced exactly via `_double_softmax_xent` | single `F.cross_entropy` on raw logits (no final softmax layer) |
| 6 | Reconstruction target is a meaningless location-ID magnitude | decoder reconstructs the raw location id (unchanged) | decoder reconstructs (lat, lon) grid coordinates |
| 7 | λ signs baked in, one shared optimizer | `LEGACY_LAMBDAS` keeps the original signed convention, one optimizer over all four sub-networks | `FIXED_LAMBDAS` are all positive; the adversarial minus sign is explicit in the encoder's loss expression; four independent optimizers |
| 8 | Privacy measured only by the in-model adversary | unchanged — out of scope for Phase 0, this is Phase 3's external-attacker suite | same — `internal_reid` in the audit output is still the in-model adversary; do not read it as a privacy guarantee |

Note on issue #5: Utkarsh's PUFFIN fork (`fairmopae.py` in both of his
directories) already uses `from_logits=False` — this specific bug did **not**
carry over into PUFFIN. `legacy` mode here still reproduces it because it is
meant to match published upstream `mopae.py`, not the PUFFIN fork.

Everything else from PUFFIN's fork (issues #2, #3, #4, #6, #8, and the shared
optimizer in #7) is present unchanged in both of Utkarsh's directories, so any
number from the rejected PUFFIN draft inherits them.

## Fairness losses (`puffin/fairness/losses.py`)

Ported from `models_mopae.py`'s `fairness=` branches, per the PI's decision
after Checkpoint 0.

- **`ed` → `ed_exact` + `ed_soft`.** `ed_exact` is a faithful port of
  `fairness="ed"`. **New finding, not in the Phase 0 audit:** the original
  `ed` loss computes the *predicted* user via `tf.argmax(y3_pred)` then
  `tf.one_hot(...)`, both gradient-free in TensorFlow. Every term in that
  loss is therefore a constant with respect to every trainable variable —
  `fairness="ed"` logged a number but contributed **zero training signal**.
  This is the *default* `fairness=` value in `Mo-PAE_og_fairness/fairmopae.py`
  — i.e. the config most likely to have produced the rejected PUFFIN draft's
  reported fairness numbers. Verified in PyTorch: `ed_exact(...).requires_grad
  == False`; `scripts/smoke_test_fairness.py` confirms this directly. This is
  an additional, independent reason (beyond issues #2–#4, #6, #8) the old
  draft's numbers shouldn't be presented, even as a pilot.
  `ed_soft` replaces the hard argmax with the predicted *distribution*
  (`probs @ entropy_table`, an expectation instead of a point estimate), so
  it has a real gradient and is usable. Wired into `fixed_train_step` as the
  Phase 4 **L_pair_threshold baseline**, off by default (`lambda4=0.0`,
  `fairness_fn=None`) so Phase 0's already-reported numbers stay reproducible.
- **`confidence` → `confidence_exact`.** Faithful port (reduce_max → `.max()`,
  the TF code's `custom_loss` threshold gate reproduced exactly, including
  that every statistic — true entropy, predicted entropy, confidence — is a
  single scalar over the *whole batch*, not per-sample; and the whole term
  returns exactly 0 whenever that batch's max entropy-gap already clears
  0.1). Reference/logging only, not wired into training — ported because it
  took under an hour, as instructed, not because it's recommended for use.
- **`matches`, `naive` — reference only, not ported.** Their original TF
  formulas (for anyone who wants them later):
  - `matches`: `loss4 = reduce_sum(onehot(argmax(y3_pred)) * y3_true * y4)` —
    also routes through `tf.argmax`/`tf.one_hot`, so it is inert for the same
    reason as `ed_exact`.
  - `naive`: `loss4 = reduce_std(y3_pred * y4)` — standard deviation of the
    predicted-distribution-weighted entropy vector; this one *is*
    differentiable (no argmax), unlike `ed`/`matches`.
  Source: `models_mopae.py` in both `/home/NETID/ud21/Mo-PAE_og_fairness/`
  and `/home/NETID/ud21/Mo-PAE__fixed_naive/` on erebor.

## Infrastructure

- PyTorch reinstalled with a CUDA 12.1 wheel (`torch==2.5.1+cu121`) matching
  the erebor driver (535.54.03, max CUDA 12.2) — the first install resolved a
  CPU-only CUDA-13 build. All four A4500s confirmed visible
  (`torch.cuda.device_count() == 4`). Done entirely inside `.venv`; the system
  driver was not touched.
- Code (not data, not results) pushed to a private GitHub remote:
  `https://github.com/ajahanba/puffin2` (private, via the `gh` CLI already
  authenticated on erebor as `ajahanba`).
- `/data` is included in erebor's system-level backups — confirmed by seeing
  `mashhadi`'s files under `/backup/dataBackup/` and a more recent
  `/backup/dataBackup09172026/` snapshot. Snapshot cadence/retention policy
  is not visible from a regular user account; ask erebor's sysadmin if that
  matters for anything irreplaceable.
