# Threat model, per mechanism (plan Phase 3.1)

What each mechanism actually releases, and which attacks in this directory
apply to it. An attack that doesn't match what a mechanism releases isn't
run against it — e.g. TUL needs released trajectories, so it's meaningless
against Mo-PAE/PUFFIN, which releases encoder features, not trajectories.

| Mechanism | Released artifact | Attacks that apply |
|---|---|---|
| Mo-PAE / PUFFIN | encoder output features `f` (worst case: the trained encoder itself, so an attacker can run arbitrary inputs through it) | fresh external re-id on `f` (`reid.py`); MIA on the utility (next-location) model trained on `f`; reconstruction attack on the decoder, if released |
| TrajGAN | synthetic trajectories | TUL on the synthetic trajectories (`tul_maintul.py`, ported from MainTUL); MIA (was a given trajectory in the generator's training set?) |
| GI-DP | perturbed (noised) location sequences | TUL on the perturbed sequences (`tul_maintul.py`); re-id directly on the perturbed sequences (same `reid.py` attacker, different input) |

Re-id risk numbers reported anywhere in this repo (Checkpoints 2/2b/2c) are
always from a **fresh external attacker** trained after the mechanism is
frozen (`puffin/attacks/reid.py`), never the in-model adversary (`id_recognizer`)
the encoder was trained to fool — that would overstate how private the
released features actually are, which is exactly issue #8 from the Phase 0
audit (`reports/00_mopae_audit.md`).

## Modules in this directory

- `windowing.py` — SL-length windows from a Phase 1 unified parquet;
  `build_sl_windows` (future/temporal split) and
  `build_sl_windows_contemporaneous` (no time gap, Checkpoint 2c).
- `reid.py` — the fresh external re-id attacker: best of {MLP, GRU} per
  seed (validation-selected), early-stopped, 5 seeds. This is Phase 3 task
  2's "finalized" attacker — see `CHANGES_FROM_ORIGINAL.md` for what
  "finalized" added (gradient-boosted trees as a third candidate
  architecture, per the plan's 3.2.1).
- `analysis.py` — OLS (plain and cluster-robust) with proper SEs/CIs,
  bootstrap CIs over users per decile, paired bootstrap tests between two
  methods on the same users.
- `mia_inhouse.py` — Phase 3 task 3, membership inference. POIPrivacy
  (https://github.com/KunlinChoi/POIPrivacy) was tried first and ruled out
  after a genuine, bounded attempt — its LocMIA/TrajMIA attacks are built
  around a specific GETNext graph-transformer POI recommender needing a
  dense O(n_pois^2) adjacency matrix and POI-category node features this
  project's data doesn't have (infeasible for GeoLife's 85,074 cells
  outright); see `CHANGES_FROM_ORIGINAL.md` for the full account.
  `mia_inhouse.py` is a plain Yeom/Shokri loss-threshold MIA against Mo-PAE-
  fixed's own encoder+next_predictor, using `mi_split` for ground truth.
- `tul_maintul.py` — Phase 3 task 4, trajectory-user linking, only ever run
  against TrajGAN/GI-DP outputs (never Mo-PAE/PUFFIN, per the table above).
  Unlike task 3's POIPrivacy, MainTUL (https://github.com/Onedean/MainTUL)
  actually ran on its own demo after two trivial compat fixes, so this
  module PORTS (does not import -- the clone in `legacy/MainTUL/` is not
  committed) its mutual-distillation LSTM-student/Transformer-teacher
  method into this project's own schema, dropping only the POI-category
  view MainTUL fuses in (this project has no POI category data, same
  reasoning as the POIPrivacy writeup below). Validated end-to-end on
  GeoLife's own real trajectories (no TrajGAN/GI-DP output exists yet to
  attack for real): see `CHANGES_FROM_ORIGINAL.md` for the full account
  and numbers.
- `adaptive.py` — Phase 3 task 5, the fairness-aware adaptive adversary.

See `../metrics/` for the per-user metrics (privacy gain, fairness,
leveling-down check) computed from these attacks' outputs.
