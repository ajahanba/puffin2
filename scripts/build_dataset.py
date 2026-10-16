#!/usr/bin/env python
"""Phase 1 driver: raw dataset -> unified schema -> filtered -> split ->
data/processed/<dataset>/{unified.parquet,stats.json}.

Usage:
    python scripts/build_dataset.py --dataset geolife \
        --config configs/data_geolife.yaml --out-dir data/processed/geolife
"""
import argparse
import json
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from puffin.data import schema, splits, filtering
from puffin.data import stats as stats_mod
from puffin.data.datasets.geolife_unified import load_geolife_unified, HAS_REAL_TIMESTAMP as GEOLIFE_HAS_TS

LOADERS = {
    "geolife": lambda cfg: (load_geolife_unified(cfg["train_csv"], cfg["test_csv"]), GEOLIFE_HAS_TS),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    df, has_ts = LOADERS[args.dataset](cfg)
    schema.validate_schema(df, allow_null_latlon=cfg.get("allow_null_latlon", False))

    df, filter_report = filtering.filter_users_by_min_observations(df, cfg.get("min_obs", 20))
    df = splits.temporal_split(df, train_frac=cfg.get("train_frac", 0.8), has_real_timestamp=has_ts)
    df = splits.membership_split(df, nonmember_traj_frac=cfg.get("nonmember_traj_frac", 0.2),
                                  user_disjoint_frac=cfg.get("user_disjoint_frac", 0.1),
                                  seed=cfg.get("seed", 0))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "unified.parquet", index=False)

    dataset_stats = stats_mod.compute_dataset_stats(df, extra=dict(
        dataset=args.dataset,
        has_real_timestamp=has_ts,
        filtering=filter_report,
        split_counts={k: int(v) for k, v in df["split"].value_counts().items()},
        mi_split_counts={k: int(v) for k, v in df["mi_split"].value_counts().items()},
        n_user_disjoint=int(df.loc[df["user_disjoint"], "user_id"].nunique()),
    ))
    stats_mod.write_stats(dataset_stats, str(out_dir / "stats.json"))
    print(json.dumps(dataset_stats, indent=2))


if __name__ == "__main__":
    main()
