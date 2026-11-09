#!/bin/bash
# GeoLife Lipschitz (3 seeds, lambda4 in {0.3,1.0}) under the current
# protocol already exists as the 10-seed geolife_lipschitz_* runs (a
# superset: more seeds, current protocol, plus lambda4=3.0). Per the
# PI's instruction (2026-11-08), using this GPU for Gowalla-SF-Bay
# Lipschitz seeds 3-4 instead (additional seeds beyond the existing
# 0,1,2 3-seed runs), both similarity bases, lambda4 in {0.3,1.0,3.0}.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/task2_formulations/gowalla_seeds34_orchestrator.log
mkdir -p logs/task2_formulations

for basis in entropy location_distribution; do
  for l4 in 0.3 1.0 3.0; do
    out_name="gowalla_sf_bay_lipschitz_${basis}_lambda${l4}_seeds34"
    out="reports/figures/task2_formulations/${out_name}"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $out_name" >> "$LOG"
      continue
    fi
    echo "$(date) LAUNCH $out_name gpu=1" >> "$LOG"
    .venv/bin/python scripts/run_phase4d_approved_batch.py \
      --dataset gowalla_sf_bay --parquet data/processed/gowalla_sf_bay/unified.parquet \
      --out-dir "$out" --mechanism lipschitz --lambda4 "$l4" --lambda-priv 0.3 \
      --similarity-basis "$basis" \
      --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
      --attack-epochs 25 --encoder-seeds 3,4 --attacker-seeds 0,1,2,3,4 \
      --device cuda:1 --no-respect-traj-boundary \
      > "logs/task2_formulations/${out_name}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "$(date) FAILED $out_name exit=$rc" >> "$LOG"
    else
      echo "$(date) OK $out_name" >> "$LOG"
    fi
  done
done
echo "$(date) gowalla seeds34 orchestrator finished" >> "$LOG"
