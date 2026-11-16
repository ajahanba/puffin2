#!/bin/bash
# Gradient-matched lipschitz_location (all 3 lambda4) + group_dro (lambda4 3.0), GPU1.
# sigma/rescale set by rule (10th-pctl of training-split pairwise distances;
# rescale matches each fairness term's mean encoder-gradient norm to
# ed_soft's at initialization, 10 batches/batch_size=512/lambda4=1.0).
DEVICE="cuda:1"
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/gradmatched_alt_objectives_gpu1_orchestrator.log
mkdir -p logs/paper reports/figures/paper

SIGMA_LOCATION=0.822244
RESCALE_LIP_LOCATION=235445.0
RESCALE_GROUP_DRO=36.67631
PARQUET=data/processed/geolife/unified.parquet
EXPECTED_SECONDS=5400

run_one() {
  out_name="$1"; shift
  out="reports/figures/paper/${out_name}"
  if [ -f "$out/summary.json" ]; then
    echo "$(date) SKIP (already done): $out_name" >> "$LOG"
    return
  fi
  rm -rf "$out"
  echo "$(date) LAUNCH $out_name device=$DEVICE args: $*" >> "$LOG"
  .venv/bin/python scripts/run_phase4d_approved_batch.py \
    --dataset geolife --parquet "$PARQUET" --out-dir "$out" \
    --sl 10 --max-windows-per-user 200 --mopae-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
    --attack-epochs 25 --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
    --device "$DEVICE" --no-respect-traj-boundary --log-gradients \
    "$@" \
    > "logs/paper/${out_name}.log" 2>&1 &
  JOB_PID=$!
  .venv/bin/python scripts/job_watchdog.py --pid $JOB_PID --expected-seconds $EXPECTED_SECONDS \
    --job-name "$out_name" --poll-seconds 120 --multiplier 3 \
    >> "logs/paper/${out_name}.watchdog.log" 2>&1 &
  WATCHDOG_PID=$!
  wait $JOB_PID
  rc=$?
  kill $WATCHDOG_PID 2>/dev/null
  if [ $rc -ne 0 ]; then
    echo "$(date) FAILED $out_name exit=$rc" >> "$LOG"
  else
    echo "$(date) OK $out_name" >> "$LOG"
  fi
}

for l4 in 0.3 1.0 3.0; do
  run_one "geolife_lipschitz_location_gradmatched_lambda${l4}_3seed" \
    --mechanism lipschitz --similarity-basis location_distribution --lambda4 "$l4" --lambda-priv 0.3 \
    --fairness-sigma "$SIGMA_LOCATION" --fairness-rescale "$RESCALE_LIP_LOCATION"
done

run_one "geolife_group_dro_gradmatched_lambda3.0_3seed" \
  --mechanism group_dro --lambda4 3.0 --lambda-priv 0.3 \
  --fairness-rescale "$RESCALE_GROUP_DRO"

echo "$(date) gradmatched_alt_objectives gpu1 orchestrator finished" >> "$LOG"
