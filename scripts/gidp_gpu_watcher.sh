#!/bin/bash
cd /data/mashhadi/puffin2
WATCHER_LOG=logs/gi_dp/gpu_handoff_watcher.log
TARGET=reports/figures/gi_dp/foursquare_gi_dp_eps0.001/summary.json
echo "$(date) watcher started, waiting for $TARGET (or its process to exit)" >> "$WATCHER_LOG"
until [ -f "$TARGET" ] || ! pgrep -f "run_gi_dp_current_protocol.py --dataset foursquare --parquet.*--epsilon 0.001" > /dev/null; do
  sleep 120
done
echo "$(date) foursquare eps=0.001 finished (or exited), killing CPU orchestrator (pid 1168027)" >> "$WATCHER_LOG"
kill -9 1168027 2>> "$WATCHER_LOG"
sleep 3
nohup bash scripts/launch_gi_dp.sh >> logs/gi_dp/orchestrator.stdout.log 2>&1 &
disown
echo "$(date) relaunched GI-DP orchestrator on GPU1, pid $!" >> "$WATCHER_LOG"
