#!/bin/bash
# Orchestrates the Phase 3b hardening grid on erebor: 3 datasets x 4 lambda4
# values, 12 runs total, batched 4-at-a-time (one per GPU) to avoid memory/
# compute contention. Resumable: skips any run whose summary.json already
# exists. Meant to be launched itself as a detached nohup process.
set -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/phase3b reports/figures/phase3b

declare -a DATASETS=("geolife" "foursquare" "gowalla_nyc")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_nyc]="data/processed/gowalla_nyc/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_nyc]="--no-respect-traj-boundary" )
LAMBDAS=(0.0 0.3 1.0 3.0)

declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
  for l4 in "${LAMBDAS[@]}"; do
    JOBS+=("${ds}:${l4}")
  done
done

n=${#JOBS[@]}
echo "$(date) orchestrator starting, $n jobs total, batches of 4" | tee -a logs/phase3b/orchestrator.log
for ((i=0; i<n; i+=4)); do
  pids=()
  for ((j=i; j<i+4 && j<n; j++)); do
    IFS=":" read -r ds l4 <<< "${JOBS[$j]}"
    out="reports/figures/phase3b/${ds}_lambda4_${l4}"
    log="logs/phase3b/${ds}_lambda4_${l4}.log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $ds lambda4=$l4" | tee -a logs/phase3b/orchestrator.log
      continue
    fi
    gpu=$(( (j - i) % 4 ))
    extra=""
    if [ "$l4" == "1.0" ]; then
      extra="--run-threat-model-check --run-mia"
    fi
    echo "$(date) launching $ds lambda4=$l4 gpu=$gpu -> $log" | tee -a logs/phase3b/orchestrator.log
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_phase3b_adaptive_robustness.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 20 --lambda-priv 0.3 --lambda4 "$l4" --attack-epochs 25 \
      --global-seeds 0,1,2,3,4 --adaptive-seeds 0,1,2,3,4 --reweight-factor 10.0 \
      --device cuda $extra > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/phase3b/orchestrator.log
done
echo "$(date) all phase3b grid jobs finished" | tee -a logs/phase3b/orchestrator.log
