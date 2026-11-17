#!/usr/bin/env python
"""Surgically tag the downward-sweep gradmatched dirs (lipschitz
entropy/location lambda4 in {0.001,0.003,0.01,0.03,0.1}, plus
group_dro lambda4=0.1 once it exists) with protocol_tag.json, same
fields/logic as tag_gradmatched_alt_objectives.py. Does NOT touch any
other directory."""
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
    f"geolife_lipschitz_{basis}_gradmatched_lambda{l4}_3seed"
    for basis in ("entropy", "location")
    for l4 in ("0.001", "0.003", "0.01", "0.03", "0.1")
] + ["geolife_group_dro_gradmatched_lambda0.1_3seed"]

for name in TARGETS:
    d = BASE / name
    if not (d / "summary.json").exists():
        print(f"SKIP (not done yet): {name}")
        continue
    if (d / "protocol_tag.json").exists():
        print(f"SKIP (already tagged): {name}")
        continue
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
