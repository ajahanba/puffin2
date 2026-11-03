#!/usr/bin/env python
"""Plateau-vs-genuine-collapse audit, per the PI's instruction (2026-11-02):
at higher privacy weights, a seed sitting at a low utility plateau may be
a REAL outcome (the privacy weight overwhelming prediction), not the
known cold-start artifact (early stopping firing during the first ~11
epochs' flat start before genuine learning kicks in). The two cannot be
told apart by epoch count alone -- a genuine collapse can also run the
full epoch budget and never leave the floor. The only way to tell them
apart is to give the seed MORE floor (double min_epochs) and see whether
it still escapes.

Flagging rule: a seed is flagged if its utility is less than half the
point's own median utility (a relative-outlier rule, not a hardcoded
constant -- works the same way regardless of dataset, mechanism, or
utility metric scale).
"""
from __future__ import annotations

import json
import pathlib
import statistics
import subprocess

REPO = pathlib.Path("/data/mashhadi/puffin2")


def flagged_seeds(summary: dict, ratio: float = 0.5) -> list:
    utils = summary["utility_per_encoder_seed"]
    med = statistics.median(utils)
    return [i for i, u in enumerate(utils) if u < ratio * med]


def _build_rerun_cmd(summary: dict, out_dir: str, dataset: str, parquet: str, seed: int,
                      doubled_min_epochs: int, device: str, sl: int = 10,
                      max_windows_per_user: int = 200, attack_epochs: int = 25,
                      traj_boundary: bool = True) -> list:
    cmd = [
        str(REPO / ".venv/bin/python"), "scripts/run_phase4d_approved_batch.py",
        "--dataset", dataset, "--parquet", parquet, "--out-dir", out_dir,
        "--mechanism", summary["mechanism"], "--lambda4", str(summary["lambda4"]),
        "--lambda-priv", str(summary.get("lambda_priv", 0.3)),
        "--similarity-basis", summary.get("similarity_basis", "entropy"),
        "--sl", str(sl), "--max-windows-per-user", str(max_windows_per_user),
        "--mopae-epochs", str(summary["max_epochs"]), "--patience", str(summary["patience"]),
        "--min-epochs", str(doubled_min_epochs), "--val-frac", str(summary["val_frac"]),
        "--attack-epochs", str(attack_epochs), "--encoder-seeds", str(seed),
        "--attacker-seeds", ",".join(str(s) for s in summary["attacker_seeds"]),
        "--device", device,
    ]
    cmd += ["--respect-traj-boundary"] if traj_boundary else ["--no-respect-traj-boundary"]
    return cmd


def audit_run(run_dir: pathlib.Path, dataset: str, parquet: str, device: str,
               traj_boundary: bool, ratio: float = 0.5) -> dict:
    """Returns a full audit record: per-seed utilities, which were
    flagged, their doubled-min_epochs rerun result, classification
    (artifact_fixed / genuine_collapse), the final per-seed array used
    for the audited mean/median, and the audited mean/median themselves.
    Never drops a genuine collapse -- it is kept in the final array."""
    summary = json.loads((run_dir / "summary.json").read_text())
    orig_utils = list(summary["utility_per_encoder_seed"])
    flagged = flagged_seeds(summary, ratio=ratio)

    seed_records = []
    final_utils = list(orig_utils)
    for i, u in enumerate(orig_utils):
        if i not in flagged:
            seed_records.append(dict(seed=i, original_utility=u, flagged=False))
            continue
        rerun_dir = run_dir.parent / "{}_auditseed{}".format(run_dir.name, i)
        doubled = summary["min_epochs"] * 2
        cmd = _build_rerun_cmd(summary, str(rerun_dir), dataset, parquet, i, doubled, device,
                                traj_boundary=traj_boundary)
        log_path = REPO / "logs/retraining_to_convergence" / "{}_auditseed{}.log".format(run_dir.name, i)
        with open(log_path, "w") as logf:
            rc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            seed_records.append(dict(seed=i, original_utility=u, flagged=True,
                                      audit_status="rerun_failed"))
            continue
        rerun_summary = json.loads((rerun_dir / "summary.json").read_text())
        rerun_u = rerun_summary["utility_per_encoder_seed"][0]
        med = statistics.median(orig_utils)
        escaped = rerun_u >= ratio * med
        classification = "artifact_fixed" if escaped else "genuine_collapse"
        final_u = rerun_u
        final_utils[i] = final_u
        seed_records.append(dict(seed=i, original_utility=u, flagged=True,
                                  doubled_min_epochs=doubled, rerun_utility=rerun_u,
                                  classification=classification, final_utility_used=final_u))

    n_collapsed = sum(1 for r in seed_records if r.get("classification") == "genuine_collapse")
    return dict(
        run=run_dir.name, n_seeds=len(orig_utils), n_flagged=len(flagged),
        n_genuine_collapse=n_collapsed, seed_records=seed_records,
        original_mean=statistics.mean(orig_utils), original_median=statistics.median(orig_utils),
        audited_mean=statistics.mean(final_utils), audited_median=statistics.median(final_utils),
        final_utils=final_utils,
    )


def apply_audit_corrections(run_dir, audit_result, out_dir=None):
    """Writes a fully-corrected per_user_04d.csv (not just a corrected
    utility scalar): for every seed classified artifact_fixed, replaces
    that seed's post_top1_encseedN / inmodel_confidence_encseedN columns
    with the doubled-min_epochs rerun's own columns (same seed label,
    verified to match). genuine_collapse seeds are left as originally
    recorded -- they are kept, not dropped, per instruction. Writes to
    out_dir (default: run_dir itself, in place) and returns the
    corrected utility array actually used."""
    import pandas as pd
    out_dir = pathlib.Path(out_dir) if out_dir else run_dir
    df = pd.read_csv(run_dir / "per_user_04d.csv")
    for rec in audit_result["seed_records"]:
        if rec.get("classification") != "artifact_fixed":
            continue
        i = rec["seed"]
        rerun_dir = run_dir.parent / "{}_auditseed{}".format(run_dir.name, i)
        rerun_df = pd.read_csv(rerun_dir / "per_user_04d.csv")
        col = f"post_top1_encseed{i}"
        assert col in rerun_df.columns, f"seed label mismatch: {col} not in rerun"
        merged = df[["user_id"]].merge(rerun_df[["user_id", col]], on="user_id", how="left", suffixes=("_old", ""))
        df[col] = merged[col].values
        inmodel_col = f"inmodel_confidence_encseed{i}"
        if inmodel_col in df.columns and inmodel_col in rerun_df.columns:
            merged2 = df[["user_id"]].merge(rerun_df[["user_id", inmodel_col]], on="user_id", how="left", suffixes=("_old", ""))
            df[inmodel_col] = merged2[inmodel_col].values

    seed_cols = sorted([c for c in df.columns if c.startswith("post_top1_encseed")],
                        key=lambda c: int(c.replace("post_top1_encseed", "")))
    df["post_top1"] = df[seed_cols].mean(axis=1)
    df["post_top1_std"] = df[seed_cols].std(axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_user_04d.csv", index=False)
    return audit_result["final_utils"]
