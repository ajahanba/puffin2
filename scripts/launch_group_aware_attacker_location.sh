#!/bin/bash
# Group-aware attacker for location-PUFFIN (lambda4=0.3) vs Mo-PAE, GPU1.
# Waits up to 2h for the pooled 10-seed location-PUFFIN lambda4=0.3 run
# (produced by scripts/launch_location_puffin_seeds39.sh on GPU0); falls
# back to the existing 3-seed run (and says so) if it isn't ready by then.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/group_aware_attacker_location_orchestrator.log
mkdir -p logs/paper reports/figures/paper

POOLED="reports/figures/paper/geolife_ed_soft_location_gradmatched_lambda0.3_10seed/summary.json"
WAITED=0
MAX_WAIT=7200
while [ ! -f "$POOLED" ] && [ "$WAITED" -lt "$MAX_WAIT" ]; do
  sleep 60
  WAITED=$((WAITED + 60))
done
if [ -f "$POOLED" ]; then
  echo "$(date) pooled 10-seed location lambda0.3 run is ready after ${WAITED}s wait" >> "$LOG"
else
  echo "$(date) pooled 10-seed location lambda0.3 run NOT ready after ${WAITED}s -- proceeding with the existing 3-seed run" >> "$LOG"
fi

echo "$(date) LAUNCH run_group_aware_attacker_location.py device=cuda:1" >> "$LOG"
.venv/bin/python scripts/run_group_aware_attacker_location.py > logs/paper/group_aware_attacker_location.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "$(date) FAILED run_group_aware_attacker_location.py exit=$rc" >> "$LOG"
  exit 1
fi
echo "$(date) OK run_group_aware_attacker_location.py" >> "$LOG"

echo "$(date) LAUNCH score_group_aware_attacker_location.py" >> "$LOG"
.venv/bin/python scripts/score_group_aware_attacker_location.py > logs/paper/score_group_aware_attacker_location.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "$(date) FAILED score_group_aware_attacker_location.py exit=$rc" >> "$LOG"
  exit 1
fi
echo "$(date) OK score_group_aware_attacker_location.py" >> "$LOG"
echo "$(date) group_aware_attacker_location orchestrator finished" >> "$LOG"
