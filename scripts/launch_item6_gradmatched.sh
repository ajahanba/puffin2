#!/bin/bash
# Item 6 redo (PI instruction, 2026-11-09): explicit --fairness-rescale
# (not a silent default), --log-gradients for per-epoch encoder-gradient
# norms (seed 0 only), distinct naming (gradmatched, not "rescaled") so
# it's never confused with the original unscaled 10-seed run at
# reports/figures/task2_formulations/geolife_ed_soft_location_distribution_lambda{0.3,1.0}_10seed
# (confirmed untouched). Each job is wrapped with the watchdog
# (scripts/job_watchdog.py): expected duration estimated from GeoLife's
# existing 10-seed ed_soft_location runs (~65-78 min for 10 seeds there;
# ~25 min for 3 seeds here is a reasonable scale-down), 3x threshold.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/item6_gradmatched_orchestrator.log
mkdir -p logs/paper reports/figures/paper reports/diagnostics

RESCALE=1.666341e7
EXPECTED_SECONDS=1800  # 30 min, conservative for a 3-seed GeoLife ed_soft run

for l4 in 0.3 1.0 3.0; do
  out_name="geolife_ed_soft_location_gradmatched_lambda${l4}_3seed"
  out="reports/figures/paper/${out_name}"
  if [ -f "$out/summary.json" ]; then
    echo "$(date) SKIP (already done): $out_name" >> "$LOG"
    continue
  fi
  echo "$(date) LAUNCH $out_name gpu=1" >> "$LOG"
  .venv/bin/python scripts/run_phase4d_approved_batch.py \
    --dataset geolife --parquet data/processed/geolife/unified.parquet \
    --out-dir "$out" --mechanism ed_soft --lambda4 "$l4" --lambda-priv 0.3 \
    --similarity-basis location_distribution --fairness-rescale "$RESCALE" --log-gradients \
    --sl 10 --max-windows-per-user 200 \
    --mopae-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
    --attack-epochs 25 --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
    --device cuda:1 --no-respect-traj-boundary \
    > "logs/paper/${out_name}.log" 2>&1 &
  JOB_PID=$!
  .venv/bin/python scripts/job_watchdog.py --pid $JOB_PID --expected-seconds $EXPECTED_SECONDS \
    --job-name "$out_name" --poll-seconds 60 --multiplier 3 \
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
done
echo "$(date) item6 gradmatched orchestrator finished" >> "$LOG"
