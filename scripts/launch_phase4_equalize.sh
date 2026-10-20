#!/bin/bash
# Equalizes the Phase 4 tuning budget: every formulation gets the same
# lambda4 grid {0.3, 1.0, 3.0} that ed_soft already had via Checkpoint 3b.
# Adds the missing points only (resumable via skip-if-done).
set -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/phase4 reports/figures/phase4

declare -a DATASETS=("geolife" "foursquare" "gowalla_nyc")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_nyc]="data/processed/gowalla_nyc/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_nyc]="--no-respect-traj-boundary" )

declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
  for f in lipschitz floor variance group_dro per_user_minimax; do
    for l4 in 0.3 3.0; do
      JOBS+=("${ds}:${f}:${l4}")
    done
  done
  JOBS+=("${ds}:group_adversarial:3.0")
done

GPUS=(0 1 2 3)
batch_size=${#GPUS[@]}
n=${#JOBS[@]}
echo "$(date) equalize orchestrator starting, $n jobs total, batches of $batch_size (GPUs ${GPUS[*]})" | tee -a logs/phase4/orchestrator_equalize.log
for ((i=0; i<n; i+=batch_size)); do
  pids=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r ds f l4 <<< "${JOBS[$j]}"
    out="reports/figures/phase4/${ds}_${f}_lambda${l4}"
    log="logs/phase4/${ds}_${f}_lambda${l4}.log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $ds $f lambda4=$l4" | tee -a logs/phase4/orchestrator_equalize.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $ds $f lambda4=$l4 gpu=$gpu -> $log" | tee -a logs/phase4/orchestrator_equalize.log
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_phase4_formulations.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" --formulation "$f" \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 20 --lambda-priv 0.3 --lambda4 "$l4" --attack-epochs 25 \
      --global-seeds 0,1,2,3,4 --adaptive-seeds 0,1,2,3,4 --reweight-factor 10.0 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/phase4/orchestrator_equalize.log
done
echo "$(date) all equalize jobs finished" | tee -a logs/phase4/orchestrator_equalize.log
