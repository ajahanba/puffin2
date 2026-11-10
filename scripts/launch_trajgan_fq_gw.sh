#!/bin/bash
# TrajGAN Foursquare -> Gowalla-SF-Bay, week-grouping rework (PI instruction,
# 2026-11-09/10). GeoLife is already done (2/3 converged, genseed2 confirmed
# non-converged under the checkpoint-aware rule). Order: Foursquare -> Gowalla,
# GPU0. Per-dataset: if all 3 generator seeds come out non-converged under
# the checkpoint/onset rule, STOP before moving to the next dataset and do
# not tune further (explicit PI instruction).
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/trajgan_fq_gw_orchestrator.log
mkdir -p logs/paper reports/figures/paper reports/diagnostics

EPOCH_BUDGET=80
EXPECTED_SECONDS=3600

declare -A PARQUET=( [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_sf_bay]="data/processed/gowalla_sf_bay/unified.parquet" )
declare -A NOPRIV=( [foursquare]="reports/figures/retraining_to_convergence/foursquare_no_privacy"
                     [gowalla_sf_bay]="reports/figures/retraining_to_convergence/gowalla_sf_bay_no_privacy" )

for ds in foursquare gowalla_sf_bay; do
  fail_count=0
  for genseed in 0 1 2; do
    out_name="${ds}_trajgan_genseed${genseed}"
    out="reports/figures/paper/${out_name}"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $out_name" >> "$LOG"
    else
      rm -rf "$out"
      echo "$(date) LAUNCH $out_name gpu=0 group_by=week" >> "$LOG"
      .venv/bin/python scripts/run_trajgan_current_protocol.py \
        --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" \
        --no-privacy-run-dir "${NOPRIV[$ds]}" \
        --max-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
        --attack-epochs 25 --trajgan-epochs $EPOCH_BUDGET --trajgan-seed $genseed \
        --trajgan-group-by week --trajgan-train-min-traj-len 2 \
        --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
        --device cuda:0 \
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
  echo "$(date) $ds: $fail_count/3 seeds non-converged" >> "$LOG"
  if [ "$fail_count" -eq 3 ]; then
    echo "$(date) ALL 3 SEEDS NON-CONVERGED for $ds under the checkpoint/onset rule -- STOPPING per PI instruction, not tuning further" >> "$LOG"
    echo "trajgan fq_gw orchestrator finished (stopped early, $ds non-converging)" >> "$LOG"
    exit 1
  fi
done
echo "$(date) trajgan fq_gw orchestrator finished" >> "$LOG"
