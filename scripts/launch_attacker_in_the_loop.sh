#!/bin/bash
# Attacker-in-the-loop fairness, rerun under the early-stopping protocol
# (2026-10-30), per the PI. NOT launched automatically by anything -- run
# manually once the retraining-to-convergence batch is done, or sooner on
# any GPU that frees up. Order: GeoLife, then Gowalla-SF-Bay, then
# Foursquare (per the PI's explicit instruction for this rerun -- note
# this is a DIFFERENT order than the original 2026-10-30 09:46 launch,
# which went GeoLife/Foursquare/Gowalla-SF-Bay and crashed on the first
# Foursquare job; see CHANGES_FROM_ORIGINAL.md).
#
# The 8 GeoLife runs completed by that earlier, fixed-40-epoch launch are
# NOT reused here and are NOT used in the paper -- this round retrains
# GeoLife too, under the same early-stopping protocol as everything else,
# for a like-for-like comparison. See PROGRESS.md for a short early-signal
# summary of those 8 runs.
#
# FAULT-TOLERANT (no `set -e`): see launch_retraining_to_convergence.sh's
# header for why -- this is the same orchestrator pattern.
#
# 3 datasets x k in {1,5} x lambda4 in {0.3,1.0} x loss_variant in
# {floor,group_dro} x 3 encoder seeds = 24 job launches, 5 attacker seeds
# each, --mopae-epochs as a ceiling with early stopping (--patience,
# --val-frac), matching every other mechanism's protocol this round.
set -uo pipefail  # NOT -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/attacker_in_the_loop reports/figures/attacker_in_the_loop

declare -a DATASETS=("geolife" "gowalla_sf_bay" "foursquare")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_sf_bay]="data/processed/gowalla_sf_bay/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_sf_bay]="--no-respect-traj-boundary" )
MAX_EPOCHS=80
PATIENCE=8
VAL_FRAC=0.1

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
declare -a FAILED=()
echo "$(date) attacker-in-the-loop orchestrator starting, $n jobs total, early-stopped "\
"(max-epochs=$MAX_EPOCHS patience=$PATIENCE val-frac=$VAL_FRAC)" | tee -a logs/attacker_in_the_loop/orchestrator.log
for ((i=0; i<n; i+=batch_size)); do
  declare -a pids=()
  declare -a pid_jobs=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r ds k l4 lv <<< "${JOBS[$j]}"
    out="reports/figures/attacker_in_the_loop/${ds}_k${k}_l4${l4}_${lv}"
    job_desc="$ds k=$k lambda4=$l4 loss_variant=$lv"
    log="logs/attacker_in_the_loop/$(basename "$out").log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $job_desc" | tee -a logs/attacker_in_the_loop/orchestrator.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $job_desc gpu=$gpu -> $log" | tee -a logs/attacker_in_the_loop/orchestrator.log
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$gpu nice -n 19 \
      python scripts/run_phase4d_approved_batch.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" \
      --mechanism attacker_in_the_loop --loss-variant "$lv" --refit-every-k-epochs "$k" --lambda4 "$l4" \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs $MAX_EPOCHS --patience $PATIENCE --val-frac $VAL_FRAC --lambda-priv 0.3 --attack-epochs 25 \
      --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
    pid_jobs+=("$job_desc")
  done
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    job_desc="${pid_jobs[$idx]}"
    if wait "$pid"; then
      echo "$(date) OK: $job_desc" | tee -a logs/attacker_in_the_loop/orchestrator.log
    else
      ec=$?
      echo "$(date) FAILED (exit $ec): $job_desc -- see its log" | tee -a logs/attacker_in_the_loop/orchestrator.log
      FAILED+=("$job_desc (exit $ec)")
    fi
  done
  echo "$(date) batch starting at index $i done" | tee -a logs/attacker_in_the_loop/orchestrator.log
done
echo "$(date) attacker-in-the-loop orchestrator finished, ${#FAILED[@]} failed of $n" \
  | tee -a logs/attacker_in_the_loop/orchestrator.log
{
  echo "Failed jobs summary ($(date)), ${#FAILED[@]} of $n:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
} > logs/attacker_in_the_loop/failed_jobs_summary.txt
cat logs/attacker_in_the_loop/failed_jobs_summary.txt
