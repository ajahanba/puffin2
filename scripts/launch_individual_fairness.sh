#!/bin/bash
# Task 1 of the fairness-reframing instruction (2026-11-02): runs on CPU,
# nice'd and thread-capped, in parallel with GPU training.
cd /data/mashhadi/puffin2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
nice -n 19 .venv/bin/python scripts/compute_individual_fairness.py \
  > logs/individual_fairness/run.log 2>&1
echo "$(date) individual fairness batch finished" >> logs/individual_fairness/run.log
