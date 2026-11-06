#!/bin/bash
# Moved to GPU1 2026-11-06 once Task 2 finished and freed it; see
# run commentary in orchestrator.log / groupdro_watcher.log for the handoff.
# GI-DP baseline, current protocol, per the PI's instruction (2026-11-03/04).
# CPU-only, thread-capped (also capped inside run_gi_dp_current_protocol.py
# itself, so this is belt-and-suspenders), fault-tolerant: no set -e, each
# job's failure is logged and the queue continues.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
LOG=logs/gi_dp/orchestrator.log
mkdir -p logs/gi_dp reports/figures/gi_dp

declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_sf_bay]="data/processed/gowalla_sf_bay/unified.parquet" )
EPSILONS="0.0005 0.001 0.002 0.005 0.01"

for ds in geolife foursquare gowalla_sf_bay; do
  for eps in $EPSILONS; do
    out="reports/figures/gi_dp/${ds}_gi_dp_eps${eps}"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $ds eps=$eps" >> "$LOG"
      continue
    fi
    echo "$(date) LAUNCH $ds eps=$eps" >> "$LOG"
    nice -n 15 .venv/bin/python scripts/run_gi_dp_current_protocol.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --epsilon "$eps" --out-dir "$out" \
      --no-privacy-run-dir "reports/figures/retraining_to_convergence/${ds}_no_privacy" \
      --max-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 --attack-epochs 25 \
      --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 --device cuda:1 \
      > "logs/gi_dp/${ds}_eps${eps}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "$(date) FAILED $ds eps=$eps exit=$rc" >> "$LOG"
    else
      echo "$(date) OK $ds eps=$eps" >> "$LOG"
    fi
  done
done
echo "$(date) GI-DP orchestrator finished" >> "$LOG"
