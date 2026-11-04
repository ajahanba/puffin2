#!/bin/bash
cd /data/mashhadi/puffin2
WATCHER_LOG=logs/task2_formulations/groupdro_watcher.log
TARGET=reports/figures/task2_formulations/gowalla_sf_bay_lipschitz_location_distribution_lambda3.0_3seed/summary.json
echo "$(date) watcher started, waiting for $TARGET" >> "$WATCHER_LOG"
until [ -f "$TARGET" ]; do
  sleep 60
done
echo "$(date) last Gowalla-SF-Bay job done, killing old task2 orchestrator (pid 1131925)" >> "$WATCHER_LOG"
kill -9 1131925 2>> "$WATCHER_LOG"
sleep 3
# Safety net: if the old orchestrator raced ahead into the first Foursquare job
# before the kill landed, remove its partial output so the relaunch redoes it cleanly.
pkill -9 -f "run_phase4d_approved_batch.py --dataset foursquare" 2>> "$WATCHER_LOG"
rm -rf reports/figures/task2_formulations/foursquare_lipschitz_entropy_lambda0.3_3seed
sleep 2
nohup .venv/bin/python scripts/launch_task2_formulations.py >> logs/task2_formulations/orchestrator.stdout.log 2>&1 &
disown
echo "$(date) relaunched task2 orchestrator with group-DRO inserted, pid $!" >> "$WATCHER_LOG"
