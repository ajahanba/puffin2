"""GeoLife sample -> unified schema (puffin/data/schema.py), for Phase 1.

Caveats specific to this dataset, logged here rather than discovered later:

- This is the "sample" bundled with Mo-PAE's GitHub repo
  (`legacy/Mo-PAE_upstream/sample_trainingdata/`), not the full public GeoLife
  GPS dataset (Microsoft download, listed separately in the plan's Phase 1
  dataset table). See `reports/00b_pets_provenance.md` for how it differs
  from what PETS 2023's tables likely used (~10% fewer unique locations).
- `lat`/`lon` in the raw CSV are already-gridded integer indices (1-33 in
  this sample), not real degrees. There is no way to recover true GPS
  coordinates or run an H3 sensitivity check (`tessellation.assign_h3_cells`)
  from this file alone -- that needs the real Microsoft download. `lat`/`lon`
  are carried through to the unified schema unchanged (still 1-33 grid
  indices) purely for traceability; `cell_id` uses the file's own `loc`
  column directly (densely remapped), not a re-derived grid, for continuity
  with Mo-PAE/PUFFIN's cell definition.
- There is no real timestamp, only categorical weekday/hour. `ts` is set to
  each row's original position within its (user, trajectory) group -- a
  monotonic surrogate, not wall-clock time. `has_real_timestamp = False`;
  callers of `puffin.data.splits.temporal_split` must pass that through.
- The raw `tid` column is a GLOBAL trajectory id shared across users (e.g.
  user 0 has tids 0-4, user 1 has tids 21-25, ...), not 1-indexed per user as
  the unified schema requires -- remapped here to be dense and local to each
  user_id.
"""
from __future__ import annotations

import pandas as pd

HAS_REAL_TIMESTAMP = False


def load_geolife_unified(train_csv: str, test_csv: str) -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)
    raw["orig_order"] = range(len(raw))

    user_dict = {old: new + 1 for new, old in enumerate(pd.unique(raw["label"]))}
    cell_dict = {old: new + 1 for new, old in enumerate(pd.unique(raw["loc"]))}

    df = pd.DataFrame()
    df["user_id"] = raw["label"].map(user_dict)
    df["cell_id"] = raw["loc"].map(cell_dict)
    df["lat"] = raw["lat"].astype(float)
    df["lon"] = raw["lon"].astype(float)
    df["weekday"] = raw["weekday"]
    df["hour"] = raw["hour"]
    df["ts"] = raw["orig_order"]  # monotonic surrogate, see module docstring

    traj_dict = {}
    for user, tid in zip(df["user_id"], raw["tid"]):
        local = traj_dict.setdefault(user, {})
        if tid not in local:
            local[tid] = len(local) + 1
    df["traj_id"] = [traj_dict[u][t] for u, t in zip(df["user_id"], raw["tid"])]

    return df[["user_id", "traj_id", "ts", "lat", "lon", "cell_id", "weekday", "hour"]]
