# PUFFIN v2

PyTorch rebuild of Mo-PAE/PUFFIN, auditing and fixing the issues in the
original TensorFlow code before any new fairness claim is trusted. See
`puffin_v2_plan.md` (PI's plan, kept outside this repo) for the full phase
breakdown, `CHANGES_FROM_ORIGINAL.md` for a line-by-line account of what
changed and why, and `reports/` for checkpoint write-ups.

Environment: `/data/mashhadi/puffin2/.venv` (Python 3.10, PyTorch) for this
repo; `/data/mashhadi/puffin2/.venv-legacy-tf` (Python 3.10, TensorFlow 2.10)
only for `legacy/Mo-PAE_upstream/`, the unmodified upstream script, kept as a
reproducibility reference but not committed here (it's its own git repo and
not ours to vendor) — re-fetch with:
```
git clone --depth 1 https://github.com/YutingZhan/Mo-PAE.git legacy/Mo-PAE_upstream
```

## Phase 0 — audit

```
source .venv/bin/activate
python scripts/run_phase0_audit.py \
  --train-csv data/raw/geolife_sample/GeoLifeBeijing_grid_train.csv \
  --test-csv  data/raw/geolife_sample/GeoLifeBeijing_grid_test.csv \
  --epochs 5 --seed 0 --out results/phase0/audit_seed0.json
```
