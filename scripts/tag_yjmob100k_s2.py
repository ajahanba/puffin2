#!/usr/bin/env python
"""Tag the 3 yjmob100k_s2_* run directories (second independent
2000-user sample, seed=1) with protocol_tag.json, same fields/logic as
scripts/backfill_protocol_tags.py (the standard, non-gradmatched
schema used for the first sample's yjmob100k_* runs), plus
sample_seed so the tag records which sample this run belongs to."""
import json
import pathlib
import subprocess

REPO = pathlib.Path("/data/mashhadi/puffin2")
BASE = REPO / "reports/figures/retraining_to_convergence"
REQUIRED_PROTOCOL = dict(patience=8, val_frac=0.1, min_epochs=20, min_encoder_seeds=1)
MIN_COMMIT = "ac55a06f2cde82a0fe785bdabc3fb2465c34b57c"
SAMPLE_SEED = 1

TARGETS = [
    "yjmob100k_s2_plain_lp0.1",
    "yjmob100k_s2_plain_lp0.2",
    "yjmob100k_s2_ed_soft_lambda1.0",
]


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


for name in TARGETS:
    d = BASE / name
    summ = json.load(open(d / "summary.json"))
    commit = summ.get("git_commit")
    inferred = commit is None
    protocol_ok = (
        summ.get("patience") == REQUIRED_PROTOCOL["patience"]
        and summ.get("val_frac") == REQUIRED_PROTOCOL["val_frac"]
        and summ.get("min_epochs") == REQUIRED_PROTOCOL["min_epochs"]
        and len(summ.get("encoder_seeds", [])) >= REQUIRED_PROTOCOL["min_encoder_seeds"]
        and is_descendant(commit, MIN_COMMIT)
    )
    tag = dict(
        git_commit=commit, git_commit_inferred=inferred,
        patience=summ.get("patience"), val_frac=summ.get("val_frac"),
        min_epochs=summ.get("min_epochs"), n_encoder_seeds=len(summ.get("encoder_seeds", [])),
        protocol_ok=protocol_ok,
        sample_seed=SAMPLE_SEED,
    )
    json.dump(tag, open(d / "protocol_tag.json", "w"), indent=2)
    print(f"{name}: protocol_ok={protocol_ok}, sample_seed={SAMPLE_SEED}")
