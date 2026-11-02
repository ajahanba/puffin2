#!/bin/bash
# Upgrades the GeoLife lambda3 reference-curve points {0.02,0.1,0.2,0.5,0.8}
# from 3 to 10 encoder seeds, per the PI's instruction (2026-11-02), on GPU1.
# lambda3=0.3 is NOT redone here -- geolife_plain_lambda0.0_10seed already
# has 10 seeds at that point. Fault-tolerant: no set -e, failures logged
# and skipped, queue continues.
cd /data/mashhadi/puffin2
LOG=logs/retraining_to_convergence/upgrade_to_10seed.log
GPU=1

for lp in 0.02 0.1 0.2 0.5 0.8; do
  OUT=reports/figures/retraining_to_convergence/geolife_plain_lp${lp}_10seed
  if [ -f "$OUT/summary.json" ]; then
    echo "$(date) SKIP (already done): lp=$lp" >> "$LOG"
    continue
  fi
  echo "$(date) LAUNCH lp=$lp gpu=$GPU" >> "$LOG"
  .venv/bin/python scripts/run_phase4d_approved_batch.py \
    --dataset geolife --parquet data/processed/geolife/unified.parquet \
    --out-dir "$OUT" --mechanism plain --lambda4 0.0 --lambda-priv "$lp" \
    --no-respect-traj-boundary --sl 10 --max-windows-per-user 200 \
    --mopae-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
    --attack-epochs 25 --encoder-seeds 0,1,2,3,4,5,6,7,8,9 \
    --attacker-seeds 0,1,2,3,4 --device cuda:$GPU \
    > "logs/retraining_to_convergence/geolife_plain_lp${lp}_10seed.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$(date) FAILED lp=$lp exit=$rc" >> "$LOG"
  else
    echo "$(date) OK lp=$lp" >> "$LOG"
  fi
done
echo "$(date) upgrade-to-10seed batch finished" >> "$LOG"
