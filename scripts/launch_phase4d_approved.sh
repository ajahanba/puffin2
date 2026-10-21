#!/bin/bash
# PI-approved batch: plain Mo-PAE + ed_soft (utility logging) + no-privacy
# reference + GeoLife's fixed floor_loss, each >=3 encoder seeds x 5
# attacker seeds, new strongest-per-user standard. Resource limits per the
# PI: capped OMP/MKL threads, nice'd, capped at 2 concurrent GPUs, batched
# (not fully parallel) so GBT-heavy phases across jobs don't pile up.
set -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/phase4d reports/figures/phase4d

declare -a DATASETS=("geolife" "foursquare" "gowalla_nyc")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_nyc]="data/processed/gowalla_nyc/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_nyc]="--no-respect-traj-boundary" )

declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
  JOBS+=("${ds}:plain:0.0")
  JOBS+=("${ds}:no_privacy:0.0")
  for l4 in 0.0 0.3 1.0 3.0; do
    JOBS+=("${ds}:ed_soft:${l4}")
  done
done
# GeoLife-only, per the PI's explicit scope for the floor_loss fix rerun
for l4 in 0.3 1.0 3.0; do
  JOBS+=("geolife:floor:${l4}")
done

GPUS=(0 1)  # capped at 2, per the PI
batch_size=${#GPUS[@]}
n=${#JOBS[@]}
echo "$(date) 04d approved-batch orchestrator starting, $n jobs total, batches of $batch_size (GPUs ${GPUS[*]})" | tee -a logs/phase4d/orchestrator.log
for ((i=0; i<n; i+=batch_size)); do
  pids=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r ds mech l4 <<< "${JOBS[$j]}"
    out="reports/figures/phase4d/${ds}_${mech}_lambda${l4}"
    log="logs/phase4d/${ds}_${mech}_lambda${l4}.log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $ds $mech lambda4=$l4" | tee -a logs/phase4d/orchestrator.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $ds $mech lambda4=$l4 gpu=$gpu -> $log" | tee -a logs/phase4d/orchestrator.log
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$gpu nice -n 10 \
      python scripts/run_phase4d_approved_batch.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" --mechanism "$mech" --lambda4 "$l4" \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 20 --lambda-priv 0.3 --attack-epochs 25 \
      --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/phase4d/orchestrator.log
done
echo "$(date) all 04d approved-batch jobs finished" | tee -a logs/phase4d/orchestrator.log
