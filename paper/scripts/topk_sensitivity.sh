#!/bin/sh
# Sensitivity of the matched-utility fairness comparison to the similar-pair threshold. Run from the repository root.
for q in 0.01 0.05 0.10 0.20; do
  python3 paper/scripts/compute_results.py --top $q --out paper/data/results_top$q.csv
  python3 paper/scripts/match_utility.py --inp paper/data/results_top$q.csv --out paper/data/matched_top$q.csv
done
