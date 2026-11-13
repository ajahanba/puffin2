#!/bin/bash
# Extends location-PUFFIN (gradient-matched, lambda4 in {0.3, 1.0}) from
# 3 to 10 encoder seeds: trains seeds 3-9 with the exact same protocol as
# scripts/launch_item6_gradmatched.sh (same rescale factor), then pools
# with the existing seeds-0-2 run into a final 10-seed directory via
# scripts/pool_location_puffin_seeds.py. lambda4=0.3 runs first so the
# group-aware-attacker job on GPU1 can start as soon as it's pooled.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/location_puffin_seeds39_orchestrator.log
mkdir -p logs/paper reports/figures/paper

RESCALE=1.666341e7
EXPECTED_SECONDS=5400

for l4 in 0.3 1.0; do
  out_name="geolife_ed_soft_location_gradmatched_lambda${l4}_seeds39"
  out="reports/figures/paper/${out_name}"
  if [ -f "$out/summary.json" ]; then
    echo "$(date) SKIP (already done): $out_name" >> "$LOG"
  else
    echo "$(date) LAUNCH $out_name device=cuda:0" >> "$LOG"
    .venv/bin/python scripts/run_phase4d_approved_batch.py \
      --dataset geolife --parquet data/processed/geolife/unified.parquet \
      --out-dir "$out" --mechanism ed_soft --lambda4 "$l4" --lambda-priv 0.3 \
      --similarity-basis location_distribution --fairness-rescale "$RESCALE" --log-gradients \
      --sl 10 --max-windows-per-user 200 \
      --mopae-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
      --attack-epochs 25 --encoder-seeds 3,4,5,6,7,8,9 --attacker-seeds 0,1,2,3,4 \
      --device cuda:0 --no-respect-traj-boundary \
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
      continue
    fi
    echo "$(date) OK $out_name" >> "$LOG"
  fi

  pooled_out="reports/figures/paper/geolife_ed_soft_location_gradmatched_lambda${l4}_10seed"
  if [ -f "$pooled_out/summary.json" ]; then
    echo "$(date) SKIP (already pooled): $pooled_out" >> "$LOG"
  else
    echo "$(date) POOLING lambda=$l4 -> $pooled_out" >> "$LOG"
    .venv/bin/python scripts/pool_location_puffin_seeds.py --lambda4 "$l4" >> "$LOG" 2>&1
  fi
done

echo "$(date) location_puffin_seeds39 orchestrator finished" >> "$LOG"
