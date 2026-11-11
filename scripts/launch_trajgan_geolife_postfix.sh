#!/bin/bash
# GeoLife TrajGAN rerun from scratch (PI instruction, 2026-11-10/11), post
# generator-fix (double-scaling + output-masking bugs fixed). Old results
# (produced by the buggy generator) moved to _invalid_pre_fix/, not reused.
# Epoch budget=80 reconfirmed valid against the new 150-epoch loss curve
# (no divergence observed at all, unlike the old buggy curve's epoch-99
# divergence) -- see reports/figures/paper/trajgan_loss_curve_geolife.json.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/trajgan_geolife_postfix_orchestrator.log
mkdir -p logs/paper reports/figures/paper reports/diagnostics

EPOCH_BUDGET=80
EXPECTED_SECONDS=5400

for genseed in 0 1 2; do
  out_name="geolife_trajgan_genseed${genseed}"
  out="reports/figures/paper/${out_name}"
  if [ -f "$out/summary.json" ]; then
    echo "$(date) SKIP (already done): $out_name" >> "$LOG"
    continue
  fi
  rm -rf "$out"
  echo "$(date) LAUNCH $out_name gpu=0 (post-fix generator)" >> "$LOG"
  .venv/bin/python scripts/run_trajgan_current_protocol.py \
    --dataset geolife --parquet data/processed/geolife/unified.parquet --out-dir "$out" \
    --no-privacy-run-dir reports/figures/retraining_to_convergence/geolife_no_privacy \
    --max-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
    --attack-epochs 25 --trajgan-epochs $EPOCH_BUDGET --trajgan-seed $genseed \
    --trajgan-group-by trajectory \
    --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
    --device cuda:0 \
    > "logs/paper/${out_name}_postfix.log" 2>&1 &
  JOB_PID=$!
  .venv/bin/python scripts/job_watchdog.py --pid $JOB_PID --expected-seconds $EXPECTED_SECONDS \
    --job-name "$out_name" --poll-seconds 120 --multiplier 3 \
    >> "logs/paper/${out_name}_postfix.watchdog.log" 2>&1 &
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
echo "$(date) geolife postfix orchestrator finished" >> "$LOG"
