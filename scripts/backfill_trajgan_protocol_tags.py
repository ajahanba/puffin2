#!/usr/bin/env python
"""Surgical protocol_tag.json backfill for the TrajGAN output directories
produced BEFORE the driver started writing its own protocol_tag.json.
Deliberately NOT using scripts/backfill_protocol_tags.py (a full rerun of
that script would overwrite the contamination_audit_passed/usable fields
separately merged onto 76 other directories -- same precedent as the
2026-11-0x gradmatched protocol-tag fix)."""
import json
import pathlib
import subprocess

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    git_commit = None

TARGETS = [
    ("reports/figures/paper/geolife_trajgan_genseed0", dict(
        trajgan_seed=0, trajgan_group_by="trajectory", train_min_traj_len=11, checkpoint_every=None,
        collapse_onset_epoch=None, checkpoint_used=None, non_converged=False,
        usable_note="predates checkpointing code; no loss_curve.json saved; converged "
                     "(2/3 GeoLife seeds converged in the original full-run attempt, this is one)")),
    ("reports/figures/paper/geolife_trajgan_genseed1", None),  # has loss_curve.json, read from it
    ("reports/figures/paper/geolife_trajgan_genseed2", None),
    ("reports/figures/paper/foursquare_trajgan_genseed0", None),
]

for out_dir_str, override in TARGETS:
    out_dir = pathlib.Path(out_dir_str)
    tag_path = out_dir / "protocol_tag.json"
    if tag_path.exists():
        print(f"SKIP (already has protocol_tag.json): {out_dir}")
        continue
    loss_curve_path = out_dir / "loss_curve.json"
    if override is not None:
        fields = dict(override)
    elif loss_curve_path.exists():
        lc = json.loads(loss_curve_path.read_text())
        non_converged = lc.get("non_converged", False)
        is_week_grouped = ("foursquare" in out_dir_str) or ("gowalla" in out_dir_str)
        fields = dict(
            trajgan_seed=lc.get("trajgan_seed"),
            trajgan_group_by="week" if is_week_grouped else "trajectory",
            train_min_traj_len=2 if is_week_grouped else 11,
            checkpoint_every=5, collapse_onset_epoch=lc.get("collapse_check", {}).get("onset_epoch"),
            checkpoint_used=lc.get("checkpoint_used"), non_converged=non_converged,
            usable_note=("non-converged: collapse onset within 5 epochs of the earliest eligible "
                         "checkpoint (epoch 30) -- no checkpoint satisfies the PI's recovery rule"
                         if non_converged else "trained under the checkpoint-aware collapse protocol"),
        )
    else:
        print(f"SKIP (no loss_curve.json, no override): {out_dir}")
        continue
    non_converged = fields.pop("non_converged")
    usable_note = fields.pop("usable_note")
    tag = dict(git_commit=git_commit, git_commit_inferred=True, **fields,
               non_converged=non_converged, protocol_ok=not non_converged, usable=not non_converged,
               usable_note=usable_note)
    tag_path.write_text(json.dumps(tag, indent=2))
    print(f"WROTE {tag_path}: non_converged={non_converged}")
