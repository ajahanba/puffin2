#!/bin/bash
cd /data/mashhadi/puffin2
LOG=logs/retraining_to_convergence/yjmob100k_orchestrator.log
WATCHER_LOG=logs/retraining_to_convergence/phase2_watcher.log
echo "$(date) watcher started, waiting for plain_lp0.1 summary.json" >> "$WATCHER_LOG"
until [ -f reports/figures/retraining_to_convergence/yjmob100k_plain_lp0.1/summary.json ]; do
  sleep 60
done
echo "$(date) plain_lp0.1 summary.json detected, killing old orchestrator (pid 1131722)" >> "$WATCHER_LOG"
kill -9 1131722 2>> "$WATCHER_LOG"
sleep 3
# Safety net: if the old orchestrator already raced ahead and started plain_lp0.2,
# kill that too so the GPU is free for the phase-2 controller.
pkill -9 -f "run_phase4d_approved_batch.py --dataset yjmob100k --parquet.*--out-dir.*yjmob100k_plain_lp0.2" 2>> "$WATCHER_LOG"
rm -rf reports/figures/retraining_to_convergence/yjmob100k_plain_lp0.2
sleep 2
nohup .venv/bin/python scripts/launch_yjmob100k_phase2.py >> logs/retraining_to_convergence/yjmob100k_phase2_orchestrator.stdout.log 2>&1 &
disown
echo "$(date) phase2 controller launched, pid $!" >> "$WATCHER_LOG"
