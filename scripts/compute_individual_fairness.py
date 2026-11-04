#!/usr/bin/env python
"""Task 1 of the fairness-reframing instruction (2026-11-02): computes
the individual-fairness metric (among similar pairs, the fraction whose
post-mechanism risk differs by less than epsilon) across all 4 similarity
definitions, all available mechanisms, for GeoLife, Foursquare, and
Gowalla-SF-Bay. Analysis only -- no new training. CPU-only by design
(see launch_individual_fairness.sh for the nice/thread-cap wrapper).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import puffin.metrics.individual_fairness as ifm

MIN_TEST_WINDOWS = 10
TRAJ_SHAPE_MAX_POINTS = {"geolife": 150, "foursquare": 150, "gowalla_sf_bay": 150}
RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
TASK2_BASE = pathlib.Path("reports/figures/task2_formulations")
OUT_BASE = pathlib.Path("reports/figures/individual_fairness")
PARQUET = {"geolife": "data/processed/geolife/unified.parquet",
           "foursquare": "data/processed/foursquare/unified.parquet",
           "gowalla_sf_bay": "data/processed/gowalla_sf_bay/unified.parquet"}

DELTA = 0.8
EPSILON = 0.2
TOP_FRAC = 0.10


def _run_dirs_for(ds, base=None):
    """Every run directory for this dataset under `base` (default
    retraining_to_convergence; pass TASK2_BASE for Task 2 runs) that is
    usable per its protocol tag."""
    base = base or RETRAIN_BASE
    runs = []
    for d in sorted(base.glob(f"{ds}_*")):
        if "CONTAMINATED" in d.name:
            continue
        if d.name.endswith("_auditseed" + d.name.rsplit("_auditseed", 1)[-1]) and "_auditseed" in d.name:
            continue  # single-seed audit rerun artifact, not a real comparison point
        if (base / f"{d.name}_corrected").exists():
            continue  # superseded by its audit-corrected version, picked up on its own iteration
        tag_path = d / "protocol_tag.json"
        if not tag_path.exists():
            continue
        tag = json.loads(tag_path.read_text())
        if not tag.get("usable"):
            continue
        runs.append(d)
    return runs


def build_similarities(ds):
    df = pd.read_parquet(PARQUET[ds])
    tr = ifm.train_only(df)
    total_locations = int(df["cell_id"].max())
    feats = ifm.compute_mobility_features(tr)
    entropy_by_user = dict(zip(feats.user_id, feats.s_unc))
    rog_by_user = dict(zip(feats.user_id, feats.rog_km))
    hist_by_user = ifm.per_user_cell_histogram(tr, total_locations)
    max_pts = TRAJ_SHAPE_MAX_POINTS[ds]
    seq_by_user = ifm.per_user_latlon_sequence(tr, max_points=max_pts)

    user_ids = sorted(entropy_by_user.keys())
    print(f"[{ds}] {len(user_ids)} users with train-split data", flush=True)

    sims = {}
    sims["entropy"] = ifm.sim_entropy(entropy_by_user, user_ids)
    print(f"[{ds}] entropy similarity done", flush=True)
    sims["location_distribution"] = ifm.sim_location_distribution(hist_by_user, user_ids)
    print(f"[{ds}] location-distribution similarity done", flush=True)
    sims["radius_of_gyration"] = ifm.sim_radius_of_gyration(rog_by_user, user_ids)
    print(f"[{ds}] radius-of-gyration similarity done", flush=True)
    sims["trajectory_shape"] = ifm.sim_trajectory_shape(seq_by_user, user_ids)
    print(f"[{ds}] trajectory-shape similarity done (max_points={max_pts})", flush=True)
    return user_ids, sims


def load_risk(run_dir, user_ids):
    """Returns (n_seeds, n_users) risk array aligned to `user_ids`, using
    only users present in this run with >= MIN_TEST_WINDOWS test windows
    (post_top1_encseedN, already averaged over attacker seeds)."""
    df = pd.read_csv(run_dir / "per_user_04d.csv")
    df = df[df.n_windows >= MIN_TEST_WINDOWS]
    seed_cols = sorted([c for c in df.columns if c.startswith("post_top1_encseed")],
                       key=lambda c: int(c.replace("post_top1_encseed", "")))
    if not seed_cols:
        return None, None
    df = df.set_index("user_id")
    valid_users = [u for u in user_ids if u in df.index]
    risk = df.loc[valid_users, seed_cols].to_numpy().T  # (n_seeds, n_users_valid)
    return valid_users, risk


def analyze_run(run_dir, user_ids, sims):
    valid_users, risk = load_risk(run_dir, user_ids)
    if risk is None or len(valid_users) < 20:
        return None
    idx_map = {u: i for i, u in enumerate(user_ids)}
    keep_idx = np.array([idx_map[u] for u in valid_users])

    result = dict(n_users=len(valid_users))
    for sim_name, full_sim in sims.items():
        sim = full_sim[np.ix_(keep_idx, keep_idx)]
        mean_risk = risk.mean(axis=0)

        i_delta, j_delta = ifm.select_similar_pairs(sim, delta=DELTA)
        i_top, j_top = ifm.select_similar_pairs(sim, top_frac=TOP_FRAC)

        rate_delta = ifm.bootstrap_if_rate(risk, i_delta, j_delta, EPSILON)
        rate_top = ifm.bootstrap_if_rate(risk, i_top, j_top, EPSILON)
        curve_delta = ifm.individual_fairness_curve(mean_risk, i_delta, j_delta)
        curve_top = ifm.individual_fairness_curve(mean_risk, i_top, j_top)
        dose = ifm.dose_response(sim, mean_risk, n_bins=10)

        result[sim_name] = dict(
            delta_0_8=rate_delta, top_10pct=rate_top,
            epsilon_curve_delta=curve_delta, epsilon_curve_top=curve_top,
            dose_response=dose,
        )
    return result


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for ds in ("geolife", "foursquare", "gowalla_sf_bay"):
        sim_cache = OUT_BASE / f"{ds}_similarities.npz"
        users_cache = OUT_BASE / f"{ds}_user_ids.json"
        if sim_cache.exists() and users_cache.exists():
            print(f"[{ds}] loading cached similarities", flush=True)
            loaded = np.load(sim_cache)
            sims = {k: loaded[k] for k in loaded.files}
            user_ids = json.loads(users_cache.read_text())
        else:
            user_ids, sims = build_similarities(ds)
            np.savez(sim_cache, **sims)
            users_cache.write_text(json.dumps(user_ids))

        runs = _run_dirs_for(ds)
        print(f"[{ds}] {len(runs)} usable run directories to analyze", flush=True)
        ds_results = {}
        for run_dir in runs:
            print(f"[{ds}] analyzing {run_dir.name}...", flush=True)
            res = analyze_run(run_dir, user_ids, sims)
            if res is not None:
                ds_results[run_dir.name] = res
        out_path = OUT_BASE / f"{ds}_individual_fairness.json"
        json.dump(ds_results, open(out_path, "w"), indent=2)
        print(f"[{ds}] wrote {out_path} ({len(ds_results)} runs)", flush=True)


if __name__ == "__main__":
    main()
