#!/bin/bash
# Orchestrates the Phase 4 formulation grid on erebor: 6 formulations x 3
# datasets, lambda4=1.0 (matching Checkpoint 3b's reference point), 18 runs
# total, batched 4-at-a-time (one per GPU). Resumable: skips any run whose
# summary.json already exists.
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
FORMULATIONS=(ed_soft lipschitz floor variance group_dro per_user_minimax group_adversarial)

declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
  for f in "${FORMULATIONS[@]}"; do
    JOBS+=("${ds}:${f}")
  done
done

GPUS=(1 2 3)  # GPU 0 is occupied by an unrelated mashhadi-owned vLLM job; avoid it, see PROGRESS.md
batch_size=${#GPUS[@]}

n=${#JOBS[@]}
echo "$(date) orchestrator starting, $n jobs total, batches of $batch_size (GPUs ${GPUS[*]})" | tee -a logs/phase4/orchestrator.log
for ((i=0; i<n; i+=batch_size)); do
  pids=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r ds f <<< "${JOBS[$j]}"
    out="reports/figures/phase4/${ds}_${f}"
    log="logs/phase4/${ds}_${f}.log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $ds $f" | tee -a logs/phase4/orchestrator.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $ds $f gpu=$gpu -> $log" | tee -a logs/phase4/orchestrator.log
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_phase4_formulations.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" --formulation "$f" \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 20 --lambda-priv 0.3 --lambda4 1.0 --attack-epochs 25 \
      --global-seeds 0,1,2,3,4 --adaptive-seeds 0,1,2,3,4 --reweight-factor 10.0 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/phase4/orchestrator.log
done
echo "$(date) all phase4 grid jobs finished" | tee -a logs/phase4/orchestrator.log
