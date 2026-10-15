"""GeoLife sample loader and two windowing/split strategies for the Phase 0 audit.

`legacy_*` reproduces mopae.py (YutingZhan/Mo-PAE) exactly, including its bugs:
windows slide over the full concatenation of all users (issue #2), the
train/test split happens after windowing (issue #3), and the target user id
for a window is taken from the window's first point rather than the point
being predicted.

`fixed_*` builds windows within a single (user, trajectory) only, splits each
user's own trajectory 80/20 by time *before* windowing, and carries weekday
/hour/lat/lon alongside location id so the model in mopae_torch.py can use a
real embedding + reconstruct grid coordinates instead of raw location-id
magnitudes (issues #2, #3, #4, #6).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class GeoLifeDataset:
    total_locations: int
    total_users: int
    entropy_by_user: dict  # new user id (1-indexed) -> normalized entropy in [0, 1]


def load_geolife_sample(train_csv: str, test_csv: str) -> pd.DataFrame:
    """Load the raw GeoLife sample CSVs and remap label/loc ids to dense 1-indexed ranges.

    Columns: label (user), tid (trajectory), lat, lon (grid coordinates, not
    real degrees, for this already-gridded sample), weekday, hour, loc (grid
    cell id).
    """
    df = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)

    id_dict = {old: new + 1 for new, old in enumerate(pd.unique(df["label"]))}
    loc_dict = {old: new + 1 for new, old in enumerate(pd.unique(df["loc"]))}
    df["label"] = df["label"].map(id_dict)
    df["loc"] = df["loc"].map(loc_dict)
    return df


def compute_user_entropy(df: pd.DataFrame) -> dict:
    """Shannon entropy of each user's location visit distribution, min-max normalized."""
    counts = df.groupby(["label", "loc"]).size().reset_index(name="count")
    probs = counts.groupby("label")["count"].apply(lambda x: x / x.sum())
    ent = probs.groupby(level=0).apply(lambda p: -float(np.sum(p * np.log2(p))))
    ent = (ent - ent.min()) / (ent.max() - ent.min())
    return ent.round(4).to_dict()


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx - 1] = 1.0
    return v


def legacy_windows(df: pd.DataFrame, context: int, total_locations: int, total_users: int):
    """Faithful port of mopae.py's get_data + non_shuffling_train_test_split.

    Windows slide over the full concatenated location sequence regardless of
    user/trajectory boundaries; a window's target user id is the id of its
    *first* point, not the user who produced the point being predicted.
    Split 80/20 happens on the already-windowed arrays, by position.
    """
    locations = df["loc"].values
    ids = df["label"].values
    n = len(locations)

    X, y, z = [], [], []
    for i in range(n - context):
        X.append(locations[i:i + context])
        y.append(_onehot(int(locations[i + context]), total_locations))
        z.append(_onehot(int(ids[i]), total_users))
    X, y, z = np.array(X, dtype=np.float32), np.array(y), np.array(z)

    split = int(0.8 * len(X)) + 1
    return (X[:split], y[:split], z[:split]), (X[split:], y[split:], z[split:])


def fixed_windows(df: pd.DataFrame, context: int, total_locations: int, total_users: int):
    """Per-(user, trajectory) windows; split each user's own timeline 80/20 first.

    Returns ((X_tr, y_tr, z_tr, wd_tr, hr_tr), (X_te, ...)) where X holds the
    `context` location ids in the window, y is the one-hot next location, z is
    the one-hot true user (always the window's own user — no ambiguity since
    windows never cross users), and wd/hr are the weekday/hour sequences
    aligned with X for the fixed model's time embeddings. Also returns lat/lon
    one step ahead (the reconstruction target) as `coord_tr`/`coord_te`,
    shape (N, context, 2).
    """
    X_tr, y_tr, z_tr, wd_tr, hr_tr, coord_tr = [], [], [], [], [], []
    X_te, y_te, z_te, wd_te, hr_te, coord_te = [], [], [], [], [], []

    for (_, _), g in df.groupby(["label", "tid"], sort=False):
        g = g.reset_index(drop=True)
        split = int(0.8 * len(g))
        parts = [(g.iloc[:split], True), (g.iloc[split:], False)]
        for part, is_train in parts:
            part = part.reset_index(drop=True)
            loc = part["loc"].values
            wd = part["weekday"].values
            hr = part["hour"].values
            lat = part["lat"].values
            lon = part["lon"].values
            user = int(part["label"].iloc[0]) if len(part) else None
            n = len(part)
            for i in range(n - context):
                tgt = X_tr if is_train else X_te
                yt = y_tr if is_train else y_te
                zt = z_tr if is_train else z_te
                wdt = wd_tr if is_train else wd_te
                hrt = hr_tr if is_train else hr_te
                ct = coord_tr if is_train else coord_te

                tgt.append(loc[i:i + context])
                yt.append(_onehot(int(loc[i + context]), total_locations))
                zt.append(_onehot(user, total_users))
                wdt.append(wd[i:i + context])
                hrt.append(hr[i:i + context])
                ct.append(np.stack([lat[i:i + context], lon[i:i + context]], axis=-1))

    train = (np.array(X_tr, dtype=np.int64), np.array(y_tr), np.array(z_tr),
             np.array(wd_tr, dtype=np.int64), np.array(hr_tr, dtype=np.int64),
             np.array(coord_tr, dtype=np.float32))
    test = (np.array(X_te, dtype=np.int64), np.array(y_te), np.array(z_te),
            np.array(wd_te, dtype=np.int64), np.array(hr_te, dtype=np.int64),
            np.array(coord_te, dtype=np.float32))
    return train, test
