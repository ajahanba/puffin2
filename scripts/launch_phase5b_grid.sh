#!/bin/bash
# Checkpoint 5b positive control: naive-vs-adaptive gap for GI-DP (epsilon
# grid) and TrajGAN, alongside plain Mo-PAE (already covered by
# Checkpoint 3b's lambda4=0.0 data, no new run needed). Runs on CPU,
# deliberately, so it doesn't compete with the Phase 4 equalization grid
# using all 4 GPUs right now. Resumable via skip-if-done.
set -e
cd /data/mashhadi/puffin2
source .venv/bin/activate
mkdir -p logs/phase5b reports/figures/phase5b

declare -a DATASETS=("geolife" "foursquare" "gowalla_nyc")
declare -A PARQUET=( [geolife]="data/processed/geolife/unified.parquet"
                     [foursquare]="data/processed/foursquare/unified.parquet"
                     [gowalla_nyc]="data/processed/gowalla_nyc/unified.parquet" )
declare -A TRAJ_FLAG=( [geolife]="--respect-traj-boundary"
                       [foursquare]="--no-respect-traj-boundary"
                       [gowalla_nyc]="--no-respect-traj-boundary" )
declare -A RAWCSV=( [geolife]="reports/figures/phase3b/geolife_lambda4_0.0/per_user_3b.csv"
                    [foursquare]="reports/figures/phase3b/foursquare_lambda4_0.0/per_user_3b.csv"
                    [gowalla_nyc]="reports/figures/phase3b/gowalla_nyc_lambda4_0.0/per_user_3b.csv" )
EPSILONS=(0.001 0.01 0.1)

declare -a JOBS=()
for ds in "${DATASETS[@]}"; do
  for eps in "${EPSILONS[@]}"; do
    JOBS+=("${ds}:gi_dp:${eps}")
  done
  JOBS+=("${ds}:trajgan:0")
done

batch_size=4
n=${#JOBS[@]}
echo "$(date) phase5b orchestrator starting, $n jobs total, batches of $batch_size, CPU" | tee -a logs/phase5b/orchestrator.log
for ((i=0; i<n; i+=batch_size)); do
  pids=()
  for ((j=i; j<i+batch_size && j<n; j++)); do
    IFS=":" read -r ds mech eps <<< "${JOBS[$j]}"
    if [ "$mech" == "gi_dp" ]; then
      out="reports/figures/phase5b/${ds}_gi_dp_eps${eps}"
      log="logs/phase5b/${ds}_gi_dp_eps${eps}.log"
      extra="--epsilon $eps"
    else
      out="reports/figures/phase5b/${ds}_trajgan"
      log="logs/phase5b/${ds}_trajgan.log"
      extra=""
    fi
    if [ -f "$out/summary.json" ]; then
      echo "$(date) SKIP (already done): $ds $mech $eps" | tee -a logs/phase5b/orchestrator.log
      continue
    fi
    echo "$(date) launching $ds $mech $eps -> $log" | tee -a logs/phase5b/orchestrator.log
    python scripts/run_phase5b_gap_mechanism.py \
      --dataset "$ds" --parquet "${PARQUET[$ds]}" --out-dir "$out" --mechanism "$mech" $extra \
      ${TRAJ_FLAG[$ds]} --sl 10 --max-windows-per-user 200 --attack-epochs 25 \
      --global-seeds 0,1,2,3,4 --adaptive-seeds 0,1,2,3,4 --reweight-factor 10.0 \
      --raw-attacker-csv "${RAWCSV[$ds]}" --device cpu > "$log" 2>&1 &
    pids+=($!)
  done
  if [ ${#pids[@]} -gt 0 ]; then
    wait "${pids[@]}"
  fi
  echo "$(date) batch starting at index $i done" | tee -a logs/phase5b/orchestrator.log
done
echo "$(date) all phase5b jobs finished" | tee -a logs/phase5b/orchestrator.log
