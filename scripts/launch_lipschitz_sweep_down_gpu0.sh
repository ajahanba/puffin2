#!/bin/bash
# Lipschitz-entropy downward lambda4 sweep, same rescale/sigma as the
# gradmatched batch, to trace the utility-risk curve up toward PUFFIN's
# region (utility >= 0.70) after the gradmatched lambda4 in {0.3,1,3}
# runs overshot (fairness gradient 10-50x the utility term's by epoch
# ~8, utility collapsed to 0.06-0.39). Shares GPU0 with the running
# YJMob100K second-sample job (GeoLife is tiny -- 181 users -- so this
# adds only a few GB; YJMob has priority if that turns out wrong).
DEVICE="cuda:0"
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/lipschitz_sweep_down_gpu0_orchestrator.log
mkdir -p logs/paper reports/figures/paper

SIGMA_ENTROPY=0.353773
RESCALE_LIP_ENTROPY=919195.5
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

for l4 in 0.001 0.003 0.01 0.03 0.1; do
  run_one "geolife_lipschitz_entropy_gradmatched_lambda${l4}_3seed" \
    --mechanism lipschitz --similarity-basis entropy --lambda4 "$l4" --lambda-priv 0.3 \
    --fairness-sigma "$SIGMA_ENTROPY" --fairness-rescale "$RESCALE_LIP_ENTROPY"
done

echo "$(date) lipschitz_sweep_down gpu0 orchestrator finished" >> "$LOG"
