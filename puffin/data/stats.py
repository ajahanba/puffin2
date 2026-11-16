"""data/processed/<dataset>/stats.json -- users, records, cells, entropy
distribution, plus the filtering/split bookkeeping so nothing about how a
dataset was built is silent."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd


def per_user_entropy(df: pd.DataFrame) -> pd.Series:
    """Vectorized: built-in groupby aggregations ("sum"), not per-group
    Python lambdas via .apply(), which keeps this tractable at
    YJMob100K's scale (100,000 users)."""
    counts = df.groupby(["user_id", "cell_id"]).size().reset_index(name="count")
    user_totals = counts.groupby("user_id")["count"].transform("sum")
    probs = counts["count"] / user_totals
    counts["entropy_term"] = -probs * np.log2(probs)
    return counts.groupby("user_id")["entropy_term"].sum()


def windows_at_sl(df: pd.DataFrame, sl: int = 10) -> dict:
    """How many usable length-`sl` windows this dataset actually yields, and
    how many users have at least one -- n_records alone doesn't say how
    much of a dataset survives windowing, which is what training actually
    consumes."""
    traj_sizes = df.groupby(["user_id", "traj_id"]).size()
    windows_per_traj = (traj_sizes - sl).clip(lower=0)
    windows_per_user = windows_per_traj.groupby(level=0).sum()
    return dict(
        sl=sl,
        total_windows=int(windows_per_traj.sum()),
        users_total=int(len(windows_per_user)),
        users_with_ge1_window=int((windows_per_user >= 1).sum()),
        location_vocab_size=int(df["cell_id"].nunique()),
    )


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
