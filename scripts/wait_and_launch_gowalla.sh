#!/bin/bash
# Waits for either the GeoLife (GPU0) or Foursquare (GPU1) postfix
# orchestrator to finish, then launches Gowalla-SF-Bay on whichever GPU
# just freed -- per the PI's instruction (2026-11-11) not to wait for
# both sequentially.
cd /data/mashhadi/puffin2
LOG=logs/paper/trajgan_gowalla_dispatch.log
echo "$(date) waiting for GeoLife (GPU0) or Foursquare (GPU1) orchestrator to finish..." >> "$LOG"
while true; do
  if ! pgrep -f "launch_trajgan_geolife_postfix.sh" > /dev/null; then
    echo "$(date) GeoLife orchestrator finished first -- dispatching Gowalla to cuda:0" >> "$LOG"
    nohup bash scripts/launch_trajgan_gowalla_postfix.sh cuda:0 > logs/paper/trajgan_gowalla_postfix_orchestrator.stdout.log 2>&1 &
    disown
    break
  fi
  if ! pgrep -f "launch_trajgan_foursquare_postfix.sh" > /dev/null; then
    echo "$(date) Foursquare orchestrator finished first -- dispatching Gowalla to cuda:1" >> "$LOG"
    nohup bash scripts/launch_trajgan_gowalla_postfix.sh cuda:1 > logs/paper/trajgan_gowalla_postfix_orchestrator.stdout.log 2>&1 &
    disown
    break
  fi
  sleep 60
done
echo "$(date) dispatcher done" >> "$LOG"
