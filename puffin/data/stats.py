"""Phase 1 task 5: data/processed/<dataset>/stats.json -- users, records,
cells, entropy distribution, plus the filtering/split bookkeeping so nothing
about how a dataset was built is silent."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd


def per_user_entropy(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby(["user_id", "cell_id"]).size().reset_index(name="count")
    probs = counts.groupby("user_id")["count"].apply(lambda x: x / x.sum())
    return probs.groupby(level=0).apply(lambda p: -float(np.sum(p * np.log2(p))))


def compute_dataset_stats(df: pd.DataFrame, extra: dict | None = None) -> dict:
    ent = per_user_entropy(df)
    stats = dict(
        n_users=int(df["user_id"].nunique()),
        n_trajectories=int(df.groupby(["user_id", "traj_id"]).ngroups),
        n_records=int(len(df)),
        n_cells=int(df["cell_id"].nunique()),
        records_per_user=dict(
            min=int(df.groupby("user_id").size().min()),
            median=float(df.groupby("user_id").size().median()),
            max=int(df.groupby("user_id").size().max()),
        ),
        entropy_distribution=dict(
            min=float(ent.min()), p10=float(ent.quantile(0.1)), median=float(ent.median()),
            p90=float(ent.quantile(0.9)), max=float(ent.max()),
        ),
    )
    if extra:
        stats.update(extra)
    return stats


def write_stats(stats: dict, out_path: str) -> None:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2))
