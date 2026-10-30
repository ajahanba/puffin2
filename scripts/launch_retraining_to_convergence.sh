#!/bin/bash
# Retraining-to-convergence batch, in the PI's priority order (2026-10-30),
# replacing the fixed-epoch-count protocol entirely: every job here uses
# early stopping on a chronological validation split (--patience, same for
# every mechanism; see run_phase4d_approved_batch.py's module docstring).
# Target: results by November 12.
#
# FAULT-TOLERANT (no `set -e`): on 2026-10-30, a single Foursquare CUDA OOM
# killed the attacker-in-the-loop orchestrator with 16/24 jobs still
# unqueued, because that script used `set -e`. This orchestrator instead:
# isolates every job, logs failures with their exit code (full error is in
# the job's own log), continues the queue regardless, and writes a failed-
# jobs summary at the end -- logs/retraining_to_convergence/failed_jobs_summary.txt.
#
# GOWALLA SUBSET: gowalla_sf_bay (2,797 users), NOT gowalla_nyc (1,517) --
# per the PI's explicit naming for this round and for attacker-in-the-loop,
# the largest/most-powered Gowalla subset is now the standard for NEW
# training going forward. 04d/04e's existing gowalla_nyc results are
# unaffected (not retrained by this script).
#
# Priority order:
#   1. plain (lambda3/lambda_priv swept -- the Pareto reference curve) +
#      no_privacy, all 3 datasets, 3 encoder seeds.
#   2. GeoLife plain vs. ed_soft, 10 encoder seeds (the dominance recheck).
#   3. ed_soft on Foursquare and Gowalla-SF-Bay, 3 encoder seeds.
#   4. floor, GeoLife only, 3 encoder seeds.
#   5. The five single-encoder-seed formulations (lipschitz, variance,
#      group_dro, per_user_minimax, group_adversarial), all 3 datasets,
#      1 seed each (matching their existing single-seed identity -- this
#      round gives them the same early-stopping protocol, not more seeds).
set -uo pipefail  # NOT -e: see above. -u/-o pipefail still catch typos/broken pipes.
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/retraining_to_convergence reports/figures/retraining_to_convergence

declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_sf_bay]="data/processed/gowalla_sf_bay/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_sf_bay]="--no-respect-traj-boundary" )
MAX_EPOCHS=80
PATIENCE=8
VAL_FRAC=0.1

# job spec: kind:dataset:mechanism:lambda4:lambda_priv:encoder_seeds
declare -a JOBS=()

# --- item 1: plain (lambda3 sweep) + no_privacy, all 3 datasets, 3 seeds ---
for ds in geolife foursquare gowalla_sf_bay; do
  for lp in 0.02 0.1 0.2 0.3 0.5 0.8; do
    JOBS+=("refcurve:${ds}:plain:0.0:${lp}:0,1,2")
  done
  JOBS+=("noprivacy:${ds}:no_privacy:0.0:0.0:0,1,2")
done

# --- item 2: GeoLife plain vs ed_soft, 10 seeds ---
JOBS+=("10seed:geolife:plain:0.0:0.3:0,1,2,3,4,5,6,7,8,9")
for l4 in 0.3 1.0 3.0; do
  JOBS+=("10seed:geolife:ed_soft:${l4}:0.3:0,1,2,3,4,5,6,7,8,9")
done

# --- item 3: ed_soft on Foursquare + Gowalla-SF-Bay, 3 seeds ---
for ds in foursquare gowalla_sf_bay; do
  for l4 in 0.3 1.0 3.0; do
    JOBS+=("edsoft:${ds}:ed_soft:${l4}:0.3:0,1,2")
  done
done

# --- item 4: floor, GeoLife only, 3 seeds ---
for l4 in 0.3 1.0 3.0; do
  JOBS+=("floor:geolife:floor:${l4}:0.3:0,1,2")
done

# --- item 5: five single-seed formulations, all 3 datasets, 1 seed ---
for ds in geolife foursquare gowalla_sf_bay; do
  for mech in lipschitz variance group_dro per_user_minimax group_adversarial; do
    for l4 in 0.3 1.0 3.0; do
      JOBS+=("singleseed:${ds}:${mech}:${l4}:0.3:0")
    done
  done
done

GPUS=(0 1)
batch_size=${#GPUS[@]}
n=${#JOBS[@]}
declare -a FAILED=()

echo "$(date) retraining-to-convergence orchestrator starting, $n jobs total (GPUs ${GPUS[*]}, "\
"max-epochs=$MAX_EPOCHS patience=$PATIENCE val-frac=$VAL_FRAC)" | tee -a logs/retraining_to_convergence/orchestrator.log

for ((i=0; i<n; i+=batch_size)); do
  declare -a pids=()
  declare -a pid_jobs=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r kind ds mech l4 lp encseeds <<< "${JOBS[$j]}"
    case "$kind" in
      refcurve)   out="reports/figures/retraining_to_convergence/${ds}_plain_lp${lp}" ;;
      noprivacy)  out="reports/figures/retraining_to_convergence/${ds}_no_privacy" ;;
      10seed)     out="reports/figures/retraining_to_convergence/${ds}_${mech}_lambda${l4}_10seed" ;;
      edsoft)     out="reports/figures/retraining_to_convergence/${ds}_ed_soft_lambda${l4}" ;;
      floor)      out="reports/figures/retraining_to_convergence/${ds}_floor_lambda${l4}" ;;
      singleseed) out="reports/figures/retraining_to_convergence/${ds}_${mech}_lambda${l4}_1seed" ;;
    esac
    job_desc="$kind $ds $mech lambda4=$l4 lambda_priv=$lp seeds=$encseeds"
    log="logs/retraining_to_convergence/$(basename "$out").log"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $job_desc" | tee -a logs/retraining_to_convergence/orchestrator.log
      continue
    fi
    gpu=${GPUS[$(( (j - i) % batch_size ))]}
    echo "$(date) launching $job_desc gpu=$gpu -> $log" | tee -a logs/retraining_to_convergence/orchestrator.log
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$gpu nice -n 10 \
      python scripts/run_phase4d_approved_batch.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" --mechanism "$mech" --lambda4 "$l4" \
      --lambda-priv "$lp" ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 \
      --mopae-epochs $MAX_EPOCHS --patience $PATIENCE --val-frac $VAL_FRAC --attack-epochs 25 \
      --encoder-seeds "$encseeds" --attacker-seeds 0,1,2,3,4 \
      --device cuda > "$log" 2>&1 &
    pids+=($!)
    pid_jobs+=("$job_desc")
  done
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    job_desc="${pid_jobs[$idx]}"
    if wait "$pid"; then
      echo "$(date) OK: $job_desc" | tee -a logs/retraining_to_convergence/orchestrator.log
    else
      ec=$?
      echo "$(date) FAILED (exit $ec): $job_desc -- see its log" | tee -a logs/retraining_to_convergence/orchestrator.log
      FAILED+=("$job_desc (exit $ec)")
    fi
  done
  echo "$(date) batch starting at index $i done" | tee -a logs/retraining_to_convergence/orchestrator.log
done

echo "$(date) retraining-to-convergence orchestrator finished, ${#FAILED[@]} failed of $n" \
  | tee -a logs/retraining_to_convergence/orchestrator.log
{
  echo "Failed jobs summary ($(date)), ${#FAILED[@]} of $n:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
} > logs/retraining_to_convergence/failed_jobs_summary.txt
cat logs/retraining_to_convergence/failed_jobs_summary.txt
