"""Individual-fairness similarity definitions and metric, per the PI's
fairness-reframing instruction (2026-11-02): this paper's actual
contribution is whether users with similar mobility receive similar
privacy protection, and whether PUFFIN (ed_soft) makes protection fairer.
Everything here is computed from the TRAINING-split rows only
(df["split"] == "train"), never test, so the similarity definitions used
to pick "similar" pairs cannot leak the held-out risk measurements they
are later compared against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from puffin.features.mobility import compute_mobility_features, _haversine_km


def train_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["split"] == "train"].copy()


def per_user_cell_histogram(df_train: pd.DataFrame, total_locations: int) -> dict:
    """user_id -> normalized visitation histogram over cell_id in
    [1, total_locations] (index 0 unused, cell ids are 1-indexed)."""
    hists = {}
    for user_id, g in df_train.groupby("user_id"):
        counts = np.bincount(g["cell_id"].to_numpy(), minlength=total_locations + 1).astype(float)
        s = counts.sum()
        hists[int(user_id)] = counts / s if s > 0 else counts
    return hists


def per_user_latlon_sequence(df_train: pd.DataFrame, max_points: int | None = None,
                              seed: int = 0) -> dict:
    """user_id -> (M,2) array of chronologically sorted (lat, lon).
    If max_points is set and a user has more points, subsamples uniformly
    at random (then re-sorts by the original chronological order) --
    used for datasets where pairwise trajectory-shape distance over the
    full sequence would be too expensive (see sim_trajectory_shape)."""
    rng = np.random.default_rng(seed)
    seqs = {}
    for user_id, g in df_train.sort_values("ts").groupby("user_id"):
        pts = g[["lat", "lon"]].to_numpy()
        if max_points is not None and len(pts) > max_points:
            idx = np.sort(rng.choice(len(pts), size=max_points, replace=False))
            pts = pts[idx]
        seqs[int(user_id)] = pts
    return seqs


def _pairwise_from_scalar(values_by_user: dict, user_ids: list) -> np.ndarray:
    """1 - normalized |v_i - v_j|, symmetric, diagonal 1.0, shape (n,n)."""
    v = np.array([values_by_user[u] for u in user_ids], dtype=float)
    diff = np.abs(v[:, None] - v[None, :])
    rng_ = diff.max()
    return 1.0 - diff / rng_ if rng_ > 0 else np.ones_like(diff)


def sim_entropy(entropy_by_user: dict, user_ids: list) -> np.ndarray:
    """Similarity (a): 1 - normalized |H_i - H_j| on Shannon entropy
    (s_unc) of the train-split visitation distribution."""
    return _pairwise_from_scalar(entropy_by_user, user_ids)


def sim_radius_of_gyration(rog_by_user: dict, user_ids: list) -> np.ndarray:
    """Similarity (d): 1 - normalized |rog_i - rog_j|."""
    return _pairwise_from_scalar(rog_by_user, user_ids)


def sim_location_distribution(hist_by_user: dict, user_ids: list) -> np.ndarray:
    """Similarity (b): 1 - Jensen-Shannon distance (base 2, in [0,1]
    already) between users' train-split visitation distributions over
    cells. Captures WHERE people go, which entropy (a measure of HOW
    EVENLY they visit, regardless of which places) misses entirely --
    two users can have identical entropy while visiting completely
    different cells."""
    n = len(user_ids)
    H = np.stack([hist_by_user[u] for u in user_ids])
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(H[i], H[j], base=2)
            d = 0.0 if np.isnan(d) else float(d)  # degenerate when both all-zero
            sim[i, j] = sim[j, i] = 1.0 - d
    return sim


def _hausdorff_km(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Symmetric Hausdorff distance (km, haversine) between two point
    sets: max over both directions of (max over one set of the min
    distance to the other set). Fully vectorized (no inner Python loop
    over point pairs) via a single (len(a), len(b)) haversine matrix."""
    lat_a, lon_a = pts_a[:, 0], pts_a[:, 1]
    lat_b, lon_b = pts_b[:, 0], pts_b[:, 1]
    D = _haversine_km(lat_a[:, None], lon_a[:, None], lat_b[None, :], lon_b[None, :])
    d_ab = D.min(axis=1).max()
    d_ba = D.min(axis=0).max()
    return float(max(d_ab, d_ba))


def sim_trajectory_shape(seq_by_user: dict, user_ids: list) -> np.ndarray:
    """Similarity (c): 1 - normalized symmetric Hausdorff distance (km)
    between users' (possibly subsampled -- see per_user_latlon_sequence)
    train-split (lat, lon) point sets. Hausdorff, not DTW: no extra
    dependency (scipy-only), order-invariant (appropriate here since we
    are comparing WHICH places were visited and how spread out they are,
    not the sequence order, which sim_entropy/location_distribution/rog
    already don't capture either), and O(n*m) per pair via one vectorized
    haversine matrix rather than DTW's O(n*m) dynamic program run in pure
    Python per pair (far slower at this pair count)."""
    n = len(user_ids)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = _hausdorff_km(seq_by_user[user_ids[i]], seq_by_user[user_ids[j]])
    rng_ = dist.max()
    return 1.0 - dist / rng_ if rng_ > 0 else np.ones((n, n))


def _upper_triangle_pairs(sim: np.ndarray):
    """Returns (i_idx, j_idx, sim_values) for all i<j pairs."""
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return iu[0], iu[1], sim[iu]


def select_similar_pairs(sim: np.ndarray, delta: float | None = None,
                          top_frac: float | None = None):
    """Returns (i_idx, j_idx) of pairs counted as 'similar', either by a
    fixed threshold (sim >= delta) or by taking the top `top_frac`
    fraction of all pairs by similarity (so different similarity
    definitions, which can have very different score distributions, are
    made comparable by selecting the same NUMBER of pairs rather than the
    same threshold value). Exactly one of delta/top_frac must be given."""
    assert (delta is None) != (top_frac is None), "give exactly one of delta, top_frac"
    i_idx, j_idx, vals = _upper_triangle_pairs(sim)
    if delta is not None:
        mask = vals >= delta
    else:
        k = max(1, int(round(top_frac * len(vals))))
        thresh = np.partition(vals, -k)[-k]
        mask = vals >= thresh
    return i_idx[mask], j_idx[mask]


def individual_fairness_rate(risk: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray,
                              epsilon: float) -> dict:
    """Among the given (already-selected similar) pairs, the fraction
    whose |risk_i - risk_j| < epsilon. risk is a per-user array aligned
    to the same user-index ordering used to build i_idx/j_idx."""
    if len(i_idx) == 0:
        return dict(rate=None, n_pairs=0)
    diffs = np.abs(risk[i_idx] - risk[j_idx])
    return dict(rate=float(np.mean(diffs < epsilon)), n_pairs=int(len(i_idx)),
                mean_abs_diff=float(diffs.mean()))


def individual_fairness_curve(risk: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray,
                               epsilons=None) -> dict:
    """individual_fairness_rate swept over a grid of epsilon values."""
    if epsilons is None:
        epsilons = np.linspace(0.0, 0.5, 26)
    diffs = np.abs(risk[i_idx] - risk[j_idx]) if len(i_idx) else np.array([])
    rates = [float(np.mean(diffs < eps)) if len(diffs) else None for eps in epsilons]
    return dict(epsilons=[float(e) for e in epsilons], rates=rates, n_pairs=int(len(i_idx)))


def dose_response(sim: np.ndarray, risk: np.ndarray, n_bins: int = 10) -> dict:
    """Bins ALL pairs by similarity into n_bins equal-width bins and
    reports each bin's mean |risk_i - risk_j| -- "do more similar users
    actually get more similar protection?" A monotonically decreasing
    mean-abs-diff as similarity rises is the expected signature of a
    mechanism that respects individual fairness; a flat or non-monotone
    curve says it doesn't, regardless of what the single-threshold rate
    says."""
    i_idx, j_idx, sim_vals = _upper_triangle_pairs(sim)
    diffs = np.abs(risk[i_idx] - risk[j_idx])
    bin_edges = np.linspace(sim_vals.min(), sim_vals.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(sim_vals, bin_edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            out.append(dict(bin=b, sim_lo=float(bin_edges[b]), sim_hi=float(bin_edges[b + 1]),
                             n_pairs=0, mean_abs_diff=None, ci_lo=None, ci_hi=None))
            continue
        d = diffs[mask]
        rng = np.random.default_rng(0)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)])
        out.append(dict(bin=b, sim_lo=float(bin_edges[b]), sim_hi=float(bin_edges[b + 1]),
                         n_pairs=int(mask.sum()), mean_abs_diff=float(d.mean()),
                         ci_lo=float(np.quantile(boots, 0.025)), ci_hi=float(np.quantile(boots, 0.975))))
    return dict(bins=out)


def bootstrap_if_rate(risk_per_seed: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray,
                       epsilon: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstraps the individual-fairness rate over BOTH encoder seeds
    (resample which seed's risk vector to use) and users (resample pairs
    with replacement), per the PI's "bootstrap CIs over encoder seeds and
    users" instruction. risk_per_seed is (n_seeds, n_users)."""
    if len(i_idx) == 0:
        return dict(rate=None, ci_lo=None, ci_hi=None, n_pairs=0)
    rng = np.random.default_rng(seed)
    n_seeds = risk_per_seed.shape[0]
    n_pairs = len(i_idx)
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, n_seeds)
        pair_sample = rng.integers(0, n_pairs, n_pairs)
        ii, jj = i_idx[pair_sample], j_idx[pair_sample]
        diffs = np.abs(risk_per_seed[s, ii] - risk_per_seed[s, jj])
        boots.append(float(np.mean(diffs < epsilon)))
    boots = np.array(boots)
    point = individual_fairness_rate(risk_per_seed.mean(axis=0), i_idx, j_idx, epsilon)["rate"]
    return dict(rate=point, ci_lo=float(np.quantile(boots, 0.025)), ci_hi=float(np.quantile(boots, 0.975)),
                n_pairs=n_pairs)
