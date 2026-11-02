#!/bin/bash
# OBSOLETE -- DO NOT RUN. Superseded by scripts/launch_retraining_to_convergence.sh.
# Moved to scripts/legacy/ 2026-11-02: this orchestrator used fixed-epoch-count
# training (no early stopping) and is not fault-tolerant (set -e; one job failure
# kills the whole batch). All of its jobs have been superseded by early-stopped,
# fault-tolerant retraining in launch_retraining_to_convergence.sh. Kept here for
# historical reference only.
# Checkpoint 04e, per the PI (2026-10-27):
# 1. Plain Mo-PAE Pareto reference CURVE: sweep lambda_priv (lambda3) --
#    {0.02, 0.1, 0.2, 0.5, 0.8}, each x3 datasets x3 encoder seeds.
#    lambda3=0.3 is NOT retrained here -- it's exactly 04d's existing
#    <dataset>_plain_lambda0.0 run, reused as the 6th curve point to
#    save compute (BUT see the epoch-count note below -- 04d's point
#    was trained at 20 epochs, these new points at 40; the reused point
#    is flagged as a seed/epoch-count mismatch in the report, not
#    silently treated as equivalent).
# 2. GeoLife 10-encoder-seed rerun of plain + ed_soft(lambda4 in
#    {0.3,1.0,3.0}), before calling ed_soft's GeoLife result "dominance".
# 2b. no_privacy retrained at the converged epoch count, 3 seeds, all 3
#     datasets -- scripts/diagnose_no_privacy_ceiling.py found ALL THREE
#     mechanisms (no_privacy, plain, ed_soft) still rising substantially
#     at epoch 20 on GeoLife (single seed): no_privacy 0.611->0.735,
#     plain 0.473->0.719, ed_soft 0.680->0.740 between epoch 20 and 40,
#     with the relative ORDERING changing (ed_soft>no_privacy>plain at
#     20; roughly tied at 40). This is not just a no_privacy tuning
#     issue -- epoch count confounds the whole batch's utility
#     comparisons via differential convergence speed. EVERY new job in
#     this orchestrator uses --mopae-epochs 40 (the point where the
#     diagnostic's curves visibly flatten) instead of 04d's 20. 04d's
#     own 21 runs are NOT retrained here (that is a much bigger
#     compute/scope decision, flagged in the report for the PI's
#     sign-off, not taken unilaterally).
#
# Resource policy (unchanged from 04d): capped OMP/MKL threads, nice'd,
# batched to respect a 2-concurrent-GPU cap, 2 free.
set -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/phase4e reports/figures/phase4e_refcurve reports/figures/phase4e_10seed

declare -a DATASETS=("geolife" "foursquare" "gowalla_nyc")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_nyc]="data/processed/gowalla_nyc/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_nyc]="--no-respect-traj-boundary" )

declare -a JOBS=()
# --- item 1: lambda3 (lambda_priv) sweep, mechanism=plain, lambda4=0, 3 encoder seeds, 40 epochs ---
for ds in "${DATASETS[@]}"; do
  for lp in 0.02 0.1 0.2 0.5 0.8; do
    JOBS+=("refcurve:${ds}:plain:0.0:${lp}:0,1,2")
  done
done
# --- item 2b: no_privacy refit at 40 epochs, 3 seeds, all 3 datasets ---
for ds in "${DATASETS[@]}"; do
  JOBS+=("refit40:${ds}:no_privacy:0.0:0.0:0,1,2")
done
# --- item 6: GeoLife 10-encoder-seed rerun, plain + ed_soft(0.3,1.0,3.0), 40 epochs ---
JOBS+=("10seed:geolife:plain:0.0:0.3:0,1,2,3,4,5,6,7,8,9")
for l4 in 0.3 1.0 3.0; do
  JOBS+=("10seed:geolife:ed_soft:${l4}:0.3:0,1,2,3,4,5,6,7,8,9")
done

GPUS=(0 1)  # restored to the usual 2-GPU cap -- the no-privacy-ceiling
            # diagnostic that previously held GPU0 has finished.
batch_size=${#GPUS[@]}
n=${#JOBS[@]}
echo "$(date) 04e orchestrator starting, $n jobs total, batch size $batch_size (GPUs ${GPUS[*]})" | tee -a logs/phase4e/orchestrator.log
for ((i=0; i<n; i+=batch_size)); do
  pids=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r kind ds mech l4 lp encseeds <<< "${JOBS[$j]}"
    if [ "$kind" == "refcurve" ]; then
      out="reports/figures/phase4e_refcurve/${ds}_plain_lp${lp}"
    elif [ "$kind" == "refit40" ]; then
      out="reports/figures/phase4e_refit40/${ds}_${mech}_40ep"
    else
      out="reports/figures/phase4e_10seed/${ds}_${mech}_lambda${l4}_10seed"
    fi
    log="logs/phase4e/$(basename "$out").log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $out" | tee -a logs/phase4e/orchestrator.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $kind $ds $mech lambda4=$l4 lambda_priv=$lp seeds=$encseeds gpu=$gpu -> $log" | tee -a logs/phase4e/orchestrator.log
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$gpu nice -n 10 \
      python scripts/run_phase4d_approved_batch.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" --mechanism "$mech" --lambda4 "$l4" \
      --lambda-priv "$lp" ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 40 --attack-epochs 25 \
      --encoder-seeds "$encseeds" --attacker-seeds 0,1,2,3,4 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/phase4e/orchestrator.log
done
echo "$(date) all 04e jobs finished" | tee -a logs/phase4e/orchestrator.log
