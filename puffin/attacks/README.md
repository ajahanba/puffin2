# Threat model, per mechanism

What each mechanism actually releases, and which attacks in this
directory apply to it.

| Mechanism | Released artifact | Attack that applies |
|---|---|---|
| Mo-PAE / PUFFIN | encoder output features `f` (worst case: the trained encoder itself, so an attacker can run arbitrary inputs through it) | fresh external re-id on `f` (`reid.py`) |
| TrajGAN | synthetic trajectories | re-id directly on the synthetic release (same `reid.py` attacker, operating on raw cell/weekday/hour windows rather than encoder features) |
| GI-DP | perturbed (noised) location sequences | re-id directly on the perturbed sequences (same `reid.py` attacker, different input) |

Re-id risk numbers reported anywhere in this repo are always from a
**fresh external attacker** trained after the mechanism is frozen
(`puffin/attacks/reid.py`), never the in-model adversary (`id_recognizer`)
the encoder was trained to fool -- that would overstate how private the
released features actually are.

## Modules in this directory

- `windowing.py` -- builds SL-length windows from a unified parquet
  (`build_sl_windows`).
- `reid.py` -- the fresh external re-id attacker: best of {MLP, GRU} per
  seed (validation-selected), early-stopped, 5 seeds, plus a gradient-
  boosted-trees variant and a strongest-per-user aggregation across all
  three architectures.
- `analysis.py` -- bootstrap confidence intervals over users per decile,
  plus a paired bootstrap test between two methods on the same users.
- `adaptive.py` -- the fairness-aware adaptive adversary, which knows the
  fairness objective and each user's entropy group, used to evaluate the
  alternative-objective formulations under a stronger threat model than
  the group-agnostic attacker above.

See `../metrics/` for the per-user metrics (privacy gain, fairness)
computed from these attacks' outputs.
