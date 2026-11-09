#!/usr/bin/env python
"""Retroactively backfills a protocol tag onto every existing run directory
under reports/figures/retraining_to_convergence/ that predates this
script's git_commit-capture patch (commit introducing this backfill).
For each dir missing 'git_commit' in summary.json, infers the commit that
was HEAD when the run completed by comparing the summary.json mtime
against `git log` timestamps on scripts/run_phase4d_approved_batch.py (the
training driver) -- best-effort, since the exact edit time can predate the
commit timestamp, but commits are ordered and monotonic so this gives the
correct commit for any run that finished after a commit landed and before
the next one did.

Writes a sibling '<out_dir>/protocol_tag.json' instead of mutating
summary.json, so this is non-destructive and re-runnable. Also flags
'protocol_ok': whether the run's recorded hyperparameters match the
paper's required protocol (see REQUIRED_PROTOCOL below) -- if False, the
run should be excluded or rerun, not silently included in a table.
"""
import datetime
import json
import pathlib
import subprocess

REPO = pathlib.Path("/data/mashhadi/puffin2")
BASE = REPO / "reports/figures/retraining_to_convergence"
TASK2_BASE = REPO / "reports/figures/task2_formulations"
GI_DP_BASE = REPO / "reports/figures/gi_dp"

REQUIRED_PROTOCOL = dict(patience=8, val_frac=0.1, min_epochs=20, min_encoder_seeds=1)
# min_commit: the first commit where the min_epochs burn-in bug was fixed.
# Any run whose inferred/recorded commit is NOT a descendant of this one
# cannot be trusted regardless of what min_epochs says it used.
MIN_COMMIT = "ac55a06f2cde82a0fe785bdabc3fb2465c34b57c"


def git_log_entries():
    out = subprocess.check_output(
        ["git", "log", "--format=%H %cI", "--", "scripts/run_phase4d_approved_batch.py"],
        cwd=REPO,
    ).decode()
    entries = []
    for line in out.strip().splitlines():
        sha, iso = line.split(" ", 1)
        entries.append((sha, datetime.datetime.fromisoformat(iso)))
    return entries  # newest first


def commit_at_time(entries, t):
    """Returns the commit that was HEAD for this file at wall-clock time t."""
    for sha, commit_time in entries:
        if commit_time <= t:
            return sha
    return None  # older than the earliest commit in range -- pre-dates our log window


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


def main():
    entries = git_log_entries()
    n_tagged, n_ok, n_bad = 0, 0, 0
    all_dirs = (sorted(BASE.iterdir())
                + (sorted(TASK2_BASE.iterdir()) if TASK2_BASE.exists() else [])
                + (sorted(GI_DP_BASE.iterdir()) if GI_DP_BASE.exists() else []))
    for d in all_dirs:
        summ_path = d / "summary.json"
        if not summ_path.exists():
            continue
        summ = json.load(open(summ_path))
        if summ.get("git_commit"):
            commit = summ["git_commit"]
            inferred = False
        else:
            mtime = datetime.datetime.fromtimestamp(summ_path.stat().st_mtime).astimezone()
            commit = commit_at_time(entries, mtime)
            inferred = True

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
        )
        json.dump(tag, open(d / "protocol_tag.json", "w"), indent=2)
        n_tagged += 1
        n_ok += protocol_ok
        n_bad += not protocol_ok
        if not protocol_ok:
            print(f"PROTOCOL MISMATCH: {d.name}: {tag}")
    print(f"\nTagged {n_tagged} run directories: {n_ok} protocol_ok, {n_bad} flagged.")


if __name__ == "__main__":
    main()
