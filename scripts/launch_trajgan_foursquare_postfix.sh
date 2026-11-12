#!/bin/bash
# Foursquare TrajGAN rerun, post generator-fix (double-scaling + output-
# masking bugs fixed, 2026-11-10/11). Runs in PARALLEL with GeoLife's
# rerun on GPU0, per the PI's instruction (2026-11-11) not to wait.
# Week-grouping (group_by=week, train_min_traj_len=2) still applies --
# that fix was orthogonal to the generator bugs and remains needed.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/trajgan_foursquare_postfix_orchestrator.log
mkdir -p logs/paper reports/figures/paper reports/diagnostics

EPOCH_BUDGET=80
EXPECTED_SECONDS=5400

fail_count=0
for genseed in 0 1 2; do
  out_name="foursquare_trajgan_genseed${genseed}"
  out="reports/figures/paper/${out_name}"
  if [ -f "$out/summary.json" ]; then
    echo "$(date) SKIP (already done): $out_name" >> "$LOG"
  else
    rm -rf "$out"
    echo "$(date) LAUNCH $out_name gpu=0 (post-fix generator, week-grouping)" >> "$LOG"
    .venv/bin/python scripts/run_trajgan_current_protocol.py \
      --dataset foursquare --parquet data/processed/foursquare/unified.parquet --out-dir "$out" \
      --no-privacy-run-dir reports/figures/retraining_to_convergence/foursquare_no_privacy \
      --max-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
      --attack-epochs 25 --trajgan-epochs $EPOCH_BUDGET --trajgan-seed $genseed \
      --trajgan-group-by week --trajgan-train-min-traj-len 2 \
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
  fi
  non_converged=$(.venv/bin/python -c "
import json, pathlib
p = pathlib.Path('$out/loss_curve.json')
if p.exists():
    print(json.loads(p.read_text()).get('non_converged', False))
else:
    print(False)
" 2>/dev/null)
  if [ "$non_converged" = "True" ]; then
    fail_count=$((fail_count + 1))
  fi
done
echo "$(date) foursquare: $fail_count/3 seeds non-converged" >> "$LOG"
if [ "$fail_count" -eq 3 ]; then
  echo "$(date) ALL 3 SEEDS NON-CONVERGED for foursquare even post-fix -- STOPPING per PI instruction" >> "$LOG"
fi
echo "$(date) foursquare postfix orchestrator finished" >> "$LOG"
