#!/bin/bash
# TrajGAN full runs (PI instruction, 2026-11-09), order: GeoLife -> Foursquare
# -> Gowalla-SF-Bay, GPU0. Fixed epoch budget=80 chosen from GeoLife's loss
# curve (stable plateau g_loss~11-57/d_loss~0.03-0.09 over epochs ~30-98,
# ~18-epoch safety margin before the discriminator-dominance divergence
# observed starting at epoch 99 -- see reports/figures/paper/
# trajgan_loss_curve_geolife.{json,pdf}), same budget reused for all three
# datasets per the PI's instruction. 3 generator seeds (one full GAN+release
# +predictor+attacker pipeline each), standard 3 predictor seeds, 5-seed x
# 3-architecture external attacker. Each job watchdog-wrapped.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=logs/paper/trajgan_full_orchestrator.log
mkdir -p logs/paper reports/figures/paper reports/diagnostics

EPOCH_BUDGET=80
EXPECTED_SECONDS=5400  # ~90 min/generator-seed estimate (80-epoch GAN ~53min + predictor + full attacker suite)

declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_sf_bay]="data/processed/gowalla_sf_bay/unified.parquet" )
declare -A NOPRIV=( [geolife]="reports/figures/retraining_to_convergence/geolife_no_privacy"
                     [foursquare]="reports/figures/retraining_to_convergence/foursquare_no_privacy"
                     [gowalla_sf_bay]="reports/figures/retraining_to_convergence/gowalla_sf_bay_no_privacy" )

for ds in geolife foursquare gowalla_sf_bay; do
  for genseed in 0 1 2; do
    out_name="${ds}_trajgan_genseed${genseed}"
    out="reports/figures/paper/${out_name}"
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $out_name" >> "$LOG"
      continue
    fi
    echo "$(date) LAUNCH $out_name gpu=0" >> "$LOG"
    .venv/bin/python scripts/run_trajgan_current_protocol.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" \
      --no-privacy-run-dir "${NOPRIV[$ds]}" \
      --max-epochs 80 --patience 8 --min-epochs 20 --val-frac 0.1 \
      --attack-epochs 25 --trajgan-epochs $EPOCH_BUDGET --trajgan-seed $genseed \
      --encoder-seeds 0,1,2 --attacker-seeds 0,1,2,3,4 \
      --device cuda:0 --profiled-epoch-seconds 6.3 \
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
  done
done
echo "$(date) trajgan full orchestrator finished" >> "$LOG"
