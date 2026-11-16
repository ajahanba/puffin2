#!/usr/bin/env python
"""Surgically tag the 9 geolife_{lipschitz_entropy,lipschitz_location,
group_dro}_gradmatched_lambda{0.3,1.0,3.0}_3seed directories with a
protocol_tag.json, same fields/logic as scripts/tag_gradmatched.py
(which did the same for the earlier ed_soft_location gradmatched
batch), plus fairness_sigma (lipschitz only -- group_dro has none).
Does NOT touch any other directory (a full backfill rerun would wipe
the separately-merged contamination_audit_passed/usable fields on the
other already-tagged directories)."""
import datetime
import json
import pathlib
import subprocess

REPO = pathlib.Path("/data/mashhadi/puffin2")
BASE = REPO / "reports/figures/paper"
REQUIRED_PROTOCOL = dict(patience=8, val_frac=0.1, min_epochs=20, min_encoder_seeds=1)
MIN_COMMIT = "ac55a06f2cde82a0fe785bdabc3fb2465c34b57c"


def is_descendant(sha, ancestor):
    if sha is None:
        return False
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", ancestor, sha], cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


TARGETS = [
    f"geolife_{key}_gradmatched_lambda{l4}_3seed"
    for key in ("lipschitz_entropy", "lipschitz_location", "group_dro")
    for l4 in ("0.3", "1.0", "3.0")
]

for name in TARGETS:
    d = BASE / name
    summ = json.load(open(d / "summary.json"))
    commit = summ.get("git_commit")
    protocol_ok = (
        summ.get("patience") == REQUIRED_PROTOCOL["patience"]
        and summ.get("val_frac") == REQUIRED_PROTOCOL["val_frac"]
        and summ.get("min_epochs") == REQUIRED_PROTOCOL["min_epochs"]
        and len(summ.get("encoder_seeds", [])) >= REQUIRED_PROTOCOL["min_encoder_seeds"]
        and is_descendant(commit, MIN_COMMIT)
    )
    tag = dict(
        git_commit=commit, git_commit_inferred=False,
        patience=summ.get("patience"), val_frac=summ.get("val_frac"),
        min_epochs=summ.get("min_epochs"), n_encoder_seeds=len(summ.get("encoder_seeds", [])),
        protocol_ok=protocol_ok,
        fairness_rescale=summ.get("fairness_rescale"),
        fairness_sigma=summ.get("fairness_sigma"),
        usable=protocol_ok,
        usable_note="current-protocol run, gradient-matched fairness rescale/sigma explicit in summary.json",
    )
    json.dump(tag, open(d / "protocol_tag.json", "w"), indent=2)
    print(f"{name}: protocol_ok={protocol_ok}, fairness_rescale={tag['fairness_rescale']}, "
          f"fairness_sigma={tag['fairness_sigma']}, usable={tag['usable']}")
