# PUFFIN

PyTorch implementation of Mo-PAE/PUFFIN, baselines (geo-indistinguishability,
TrajGAN), and the evaluation/analysis pipeline used in the paper.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

## 1. Build the datasets

```
python scripts/build_dataset.py --dataset geolife     --config configs/data_geolife.yaml     --out-dir data/processed/geolife
python scripts/build_dataset.py --dataset foursquare  --config configs/data_foursquare.yaml  --out-dir data/processed/foursquare
python scripts/build_dataset.py --dataset gowalla      --config configs/data_gowalla_sf_bay.yaml --out-dir data/processed/gowalla_sf_bay
python scripts/build_dataset.py --dataset yjmob100k    --config configs/data_yjmob100k_full.yaml --out-dir data/processed/yjmob100k_full
```

Each run writes `data/processed/<dataset>/{unified.parquet,stats.json}`.

### YJMob100K sampling

YJMob100K's full 100K-user set is subsampled to 2000 users before any
mechanism is trained on it:

```
python scripts/sample_yjmob100k.py
python scripts/compute_yjmob100k_similarities.py
```

Writes `reports/figures/yjmob100k_sample/{unified_sample.parquet,user_ids.json,...}`.

## 2. Train the mechanisms

All mechanisms (PUFFIN, Mo-PAE, and the no-privacy/matched-utility
reference points) go through one driver, under one training protocol, so
comparisons across mechanisms are never confounded by different training
code:

```
python scripts/train_mechanism.py --dataset geolife --parquet data/processed/geolife/unified.parquet \
    --out-dir reports/figures/retraining_to_convergence/geolife_ed_soft_lambda1.0_10seed \
    --mechanism ed_soft --lambda4 1.0 --encoder-seeds 0,1,2,3,4,5,6,7,8,9
```

`--mechanism plain` (Mo-PAE) and `--mechanism no_privacy` (utility ceiling)
produce the base-table reference runs; `--mechanism ed_soft` with
`--lambda4 > 0` produces PUFFIN. The resulting `summary.json` and
`per_user_04d.csv` in `--out-dir` are the input to every table/figure below.

Location-similarity PUFFIN (`--similarity-basis location_distribution`) and
its per-term gradient check:

```
python scripts/gradient_check_location_puffin.py
```

### Alternative-objective table

Evaluates each fairness formulation (Lipschitz, floor, variance, group-DRO,
per-user minimax, group-adversarial) against the adaptive, fairness-aware
attacker at matched utility:

```
python scripts/train_alternative_objectives.py --dataset geolife --parquet data/processed/geolife/unified.parquet \
    --out-dir reports/figures/alternative_objectives/geolife_lipschitz_lambda1.0 \
    --formulation lipschitz --lambda4 1.0
```

### Baselines: geo-indistinguishability and TrajGAN

```
python scripts/run_gi_dp.py --dataset geolife --parquet data/processed/geolife/unified.parquet \
    --out-dir reports/figures/gi_dp/geolife_eps0.1 --epsilon 0.1

python scripts/run_trajgan.py --dataset geolife --parquet data/processed/geolife/unified.parquet \
    --out-dir reports/figures/trajgan/geolife
```

Both mirror `train_mechanism.py`'s output shape, so every analysis script
below works on their output unchanged.

## 3. Bootstrap CIs and comparison tables

Base table and matched-utility table, with paired bootstrap confidence
intervals:

```
python scripts/run_bootstrap_cis.py
python scripts/run_bootstrap_cis_base_table.py
python scripts/run_bootstrap_cis_yjmob100k.py
```

(`bootstrap_cis.py` holds the shared paired-bootstrap machinery these three
scripts call; it is not run directly.) Writes
`reports/figures/paper/bootstrap_cis_matched_utility.csv`.

In-model vs. external-attacker risk comparison:

```
python scripts/run_inmodel_vs_external.py
```

Writes `reports/figures/paper/inmodel_vs_external.csv`.

Group-aware (per-decile-restricted) attacker:

```
python scripts/run_group_aware_attacker.py
python scripts/score_group_aware_attacker.py
```

Writes `reports/figures/paper/group_aware_attacker.csv` (the scored,
paired-bootstrap comparison used in the paper;
`group_aware_attacker_direct.csv` is an intermediate, unscored output).

## 4. Individual fairness

Similarity-pair metrics (entropy, location-distribution, trajectory-shape,
radius-of-gyration similarity bases) and the individual-fairness rate:

```
python scripts/compute_individual_fairness.py
python scripts/generate_individual_fairness_report.py
```

Writes `reports/figures/individual_fairness/*.json` and
`reports/06_individual_fairness.md`.

## 5. Paper figures and tables

```
python paper/scripts/fig_decile_risk.py
python paper/scripts/fig_tradeoff.py
python paper/scripts/ci_marks.py
```

These read `paper/data/*.csv` and the `reports/figures/*` outputs produced
above, and write the figures/table rows used directly in the paper.

### Core results tables and the matched-utility comparison

```
python paper/scripts/compute_results.py          # -> paper/data/results_main.csv
python paper/scripts/match_utility.py             # -> paper/data/matched.csv
python paper/scripts/compute_yjmob100k.py         # -> paper/data/results_yjmob.csv
bash paper/scripts/topk_sensitivity.sh            # -> paper/data/{results,matched}_top{0.01,0.05,0.10,0.20}.csv
```

`compute_results.py` computes per-run utility, risk, and individual-fairness
measures (D_abs, D_rel) for GeoLife, Foursquare, and Gowalla from the
`reports/figures/{retraining_to_convergence,alternative_objectives,gi_dp,paper}`
outputs produced in steps 2-3 above, including TrajGAN's three
`reports/figures/paper/<dataset>_trajgan_genseed{0,1,2}/` runs, pooled
across generator seeds into a single TrajGAN row per dataset;
`compute_yjmob100k.py` does the same for YJMob100K. `match_utility.py`
finds, for each method, the point on the
Mo-PAE utility curve with matched utility (interpolated, never
extrapolated) and reports the risk/fairness comparison at that point.
`topk_sensitivity.sh` reruns both at similar-pair thresholds
top={1%,5%,10%,20%} to check the comparison's sensitivity to that choice.

Together these back Tables 1-4 and 7, Figures 2-3, and the matched-utility
comparisons reported throughout the paper.

## Tests

```
pytest tests/
```
