#!/bin/bash
# Attacker-in-the-loop fairness, per the PI's original request (queued
# since the floor_loss/in-model-confidence finding), launched NOW at
# LOWER PRIORITY per the PI's explicit instruction ("launch
# attacker-in-the-loop at lower priority, evaluated against the item-1
# curve"), after Checkpoint 04e's correction round (reports/04e_reference_curve.md).
#
# 3 datasets (GeoLife, Foursquare, Gowalla-SF-BAY -- NOT gowalla_nyc,
# per the PI's explicit naming for this formulation, a different
# Gowalla subset than every other phase in this project has used) x
# k in {1,5} x lambda4 in {0.3,1.0} x loss_variant in {floor,group_dro}
# x 3 encoder seeds = 24 job launches (each training 3 seeds
# internally), 5 attacker seeds each.
#
# Uses --mopae-epochs 40, NOT 04d's original 20 -- Checkpoint 04e found
# every mechanism undertrained at 20 epochs with no early stopping
# anywhere in the driver; there is no reason to repeat that mistake on
# a brand-new formulation.
#
# LOWER PRIORITY: nice -n 19 (vs. the usual nice -n 10), reduced thread
# count (6, vs. the usual 16), matching Phase 5b's established "low
# priority" convention -- this is explicitly less urgent than any
# future correction work, not a resource hog on a shared machine.
set -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/attacker_in_the_loop reports/figures/attacker_in_the_loop

declare -a DATASETS=("geolife" "foursquare" "gowalla_sf_bay")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_sf_bay]="data/processed/gowalla_sf_bay/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_sf_bay]="--no-respect-traj-boundary" )

declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
  for k in 1 5; do
    for l4 in 0.3 1.0; do
      for lv in floor group_dro; do
        JOBS+=("${ds}:${k}:${l4}:${lv}")
      done
    done
  done
done

GPUS=(0 1)
batch_size=${#GPUS[@]}
n=${#JOBS[@]}
echo "$(date) attacker-in-the-loop orchestrator starting, $n jobs total, LOW PRIORITY (nice 19)" | tee -a logs/attacker_in_the_loop/orchestrator.log
for ((i=0; i<n; i+=batch_size)); do
  pids=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r ds k l4 lv <<< "${JOBS[$j]}"
    out="reports/figures/attacker_in_the_loop/${ds}_k${k}_l4${l4}_${lv}"
    log="logs/attacker_in_the_loop/$(basename "$out").log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $out" | tee -a logs/attacker_in_the_loop/orchestrator.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $ds k=$k lambda4=$l4 loss_variant=$lv gpu=$gpu -> $log" | tee -a logs/attacker_in_the_loop/orchestrator.log
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=$gpu nice -n 19 \
      python scripts/run_phase4d_approved_batch.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" \
      --mechanism attacker_in_the_loop --loss-variant "$lv" --refit-every-k-epochs "$k" --lambda4 "$l4" \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 40 --lambda-priv 0.3 --attack-epochs 25 \
      --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/attacker_in_the_loop/orchestrator.log
done
echo "$(date) all attacker-in-the-loop jobs finished" | tee -a logs/attacker_in_the_loop/orchestrator.log
