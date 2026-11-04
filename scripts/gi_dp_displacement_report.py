#!/usr/bin/env python
"""GI-DP epsilon grid: per-epsilon mean displacement and cell-change rate,
per the PI's instruction (2026-11-03/04). epsilon in {0.0005, 0.001,
0.002, 0.005, 0.01} on the 0.01deg grid (mean displacement ~4km to 200m).
No training yet -- just the perturbation's own diagnostic stats.
"""
import sys
sys.path.insert(0, ".")
import json
import pathlib

import numpy as np
import pandas as pd

from puffin.baselines.gi_dp import run_gi_dp
from puffin.data.tessellation import assign_grid_cells
from puffin.features.mobility import _haversine_km

EPSILONS = [0.0005, 0.001, 0.002, 0.005, 0.01]
DATASETS = {
    "geolife": "data/processed/geolife/unified.parquet",
    "foursquare": "data/processed/foursquare/unified.parquet",
    "gowalla_sf_bay": "data/processed/gowalla_sf_bay/unified.parquet",
}
SEED = 0


def main():
    out = {}
    for ds, path in DATASETS.items():
        df = pd.read_parquet(path)
        df = df[df["lat"].notna() & df["lon"].notna()].copy()
        if len(df) > 2_000_000:
            df = df.sample(n=2_000_000, random_state=SEED)
        print(f"[{ds}] {len(df)} rows with real lat/lon", flush=True)
        # Fix (2026-11-04): re-tessellate the ORIGINAL data once here and
        # reuse its exact cell_id mapping for every perturbed copy below --
        # without this, cell_id is independently (and differently)
        # remapped each call, making "did the cell change" meaningless
        # (see tessellation.py's updated docstring for the bug this fixes).
        df = assign_grid_cells(df, resolution_deg=0.01)
        original_mapping = df.attrs["grid_cell_mapping"]
        ds_out = {}
        for eps in EPSILONS:
            perturbed = run_gi_dp(df, epsilon=eps, seed=SEED, cell_mapping=original_mapping)
            disp_km = _haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                                     perturbed["lat"].to_numpy(), perturbed["lon"].to_numpy())
            cell_change_rate = float((perturbed["cell_id"].to_numpy() != df["cell_id"].to_numpy()).mean())
            ds_out[str(eps)] = dict(
                mean_displacement_km=float(disp_km.mean()),
                median_displacement_km=float(np.median(disp_km)),
                theoretical_mean_displacement_km=2.0 / eps / 1000.0,
                cell_change_rate=cell_change_rate,
            )
            print(f"  eps={eps}: mean_disp={disp_km.mean():.3f}km (theory {2.0/eps/1000:.3f}km) "
                  f"cell_change_rate={cell_change_rate:.4f}", flush=True)
        out[ds] = ds_out
    out_path = pathlib.Path("reports/figures/gi_dp_analysis")
    out_path.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path / "epsilon_displacement_report.json", "w"), indent=2)
    print(f"wrote {out_path / 'epsilon_displacement_report.json'}")


if __name__ == "__main__":
    main()
