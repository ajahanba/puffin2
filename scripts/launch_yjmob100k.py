#!/usr/bin/env python
"""YJMob100K training, 2000-user sample, per the PI's instruction
(2026-11-03). GPU0 (GPU1 keeps running the Lipschitz queue). Same
protocol as everything else this round, with the plateau audit run
INLINE after each job (not post-hoc): any flagged seed is immediately
rerun with min_epochs doubled and merged in before moving to the next
job, so no contaminated point is ever used for a downstream decision.

Job list: no_privacy, plain Mo-PAE lambda3 in {0.02,0.1,0.2,0.3,0.5,0.8},
PUFFIN (ed_soft) lambda4 in {0.3,1.0,3.0}. 3 encoder seeds each. The
lambda3 sweep is NOT extended here even though PUFFIN's utility range is
unknown yet -- extension (if needed) is a follow-up decision once this
batch's numbers are in, same as how the GeoLife sweep itself started
simple and only extended afterward.
"""
import datetime
import json
import pathlib
import subprocess

import sys
sys.path.insert(0, ".")
from puffin.eval.plateau_audit import audit_run, apply_audit_corrections

REPO = pathlib.Path("/data/mashhadi/puffin2")
DATASET = "yjmob100k"
PARQUET = "data/processed/yjmob100k_full/unified.parquet"
OUT_BASE = REPO / "reports/figures/retraining_to_convergence"
LOG_PATH = REPO / "logs/retraining_to_convergence/yjmob100k_orchestrator.log"
N_SEEDS = 3
SAMPLE_MANIFEST = REPO / "reports/figures/yjmob100k_sample/sample_manifest.json"
GPU = 0


def log(msg):
    ts = datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def sample_users():
    return json.loads(SAMPLE_MANIFEST.read_text())["user_ids"]


def filter_to_sample(parquet_path, user_ids, out_path):
    """Remaps user_id to a dense 1..N range before saving. Without this,
    main()'s total_users = df["user_id"].max() stays at the FULL
    dataset's 100,000 even though only 2000 users are present -- the
    attacker's classification head then has a 100,000-wide output layer
    for 2000 actual users, and _batched_forward's final concatenated
    logits tensor (chunked on the INPUT, not the output width) OOMs
    (14.88 GiB single allocation, hit in practice on lambda4=1.0/3.0).
    The remap must happen once, consistently, before ANY job in this
    sample trains, so every run's user_id numbering agrees."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    sub = df[df["user_id"].isin(user_ids)].copy()
    sorted_ids = sorted(user_ids)
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted_ids, start=1)}
    sub["user_id"] = sub["user_id"].map(remap)
    sub.to_parquet(out_path, index=False)
    (out_path.parent / "user_id_remap.json").write_text(
        __import__("json").dumps({str(k): v for k, v in remap.items()}, indent=2))
    return out_path


def build_cmd(mechanism, lambda4, lambda_priv, out_name, seeds, sample_parquet):
    seed_str = ",".join(str(s) for s in seeds)
    return [
        str(REPO / ".venv/bin/python"), "scripts/run_phase4d_approved_batch.py",
        "--dataset", DATASET, "--parquet", str(sample_parquet),
        "--out-dir", str(OUT_BASE / out_name), "--mechanism", mechanism,
        "--lambda4", str(lambda4), "--lambda-priv", str(lambda_priv),
        "--sl", "10", "--max-windows-per-user", "200",
        "--mopae-epochs", "80", "--patience", "8", "--min-epochs", "20", "--val-frac", "0.1",
        "--attack-epochs", "25", "--encoder-seeds", seed_str, "--attacker-seeds", "0,1,2,3,4",
        "--device", f"cuda:{GPU}", "--no-respect-traj-boundary",
    ]


def run_job_with_audit(out_name, mechanism, lambda4, lambda_priv, sample_parquet):
    out_dir = OUT_BASE / out_name
    if (out_dir / "summary.json").exists():
        log(f"SKIP (already done): {out_name}")
        return
    cmd = build_cmd(mechanism, lambda4, lambda_priv, out_name, range(N_SEEDS), sample_parquet)
    log(f"LAUNCH {out_name} gpu={GPU}")
    job_log = REPO / f"logs/retraining_to_convergence/{out_name}.log"
    with open(job_log, "w") as logf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f"FAILED {out_name} exit={rc} -- see {job_log}")
        return
    log(f"OK {out_name}")

    audit = audit_run(out_dir, DATASET, str(sample_parquet), f"cuda:{GPU}", traj_boundary=False)
    if audit["n_flagged"] == 0:
        log(f"AUDIT {out_name}: 0 flagged seeds, clean")
        return
    log(f"AUDIT {out_name}: {audit['n_flagged']} flagged, {audit['n_genuine_collapse']} genuine collapse "
        f"-- naive_mean={audit['original_mean']:.4f} audited_mean={audit['audited_mean']:.4f}")
    corrected_dir = OUT_BASE / f"{out_name}_corrected"
    import shutil
    corrected_dir.mkdir(exist_ok=True)
    shutil.copy(out_dir / "summary.json", corrected_dir / "summary.json")
    final_utils = apply_audit_corrections(out_dir, audit, out_dir=corrected_dir)
    s = json.loads((corrected_dir / "summary.json").read_text())
    s["utility_per_encoder_seed"] = final_utils
    s["utility_mean"] = sum(final_utils) / len(final_utils)
    s["corrected_from"] = out_name
    s["plateau_audit"] = audit
    (corrected_dir / "summary.json").write_text(json.dumps(s, indent=2, default=float))
    log(f"wrote {corrected_dir.name}")


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    users = sample_users()
    log(f"{len(users)} sample users loaded from manifest")
    sample_parquet = REPO / "reports/figures/yjmob100k_sample/unified_sample.parquet"
    if not sample_parquet.exists():
        log("filtering full unified.parquet down to the 2000-user sample...")
        filter_to_sample(REPO / PARQUET, users, sample_parquet)
    log(f"using sample parquet: {sample_parquet}")

    jobs = [("yjmob100k_no_privacy", "no_privacy", 0.0, 0.0)]
    for lp in (0.02, 0.1, 0.2, 0.3, 0.5, 0.8):
        jobs.append((f"yjmob100k_plain_lp{lp}", "plain", 0.0, lp))
    for l4 in (0.3, 1.0, 3.0):
        jobs.append((f"yjmob100k_ed_soft_lambda{l4}", "ed_soft", l4, 0.3))

    log(f"{len(jobs)} YJMob100K jobs queued")
    for out_name, mech, l4, lp in jobs:
        run_job_with_audit(out_name, mech, l4, lp, sample_parquet)
    log("YJMob100K orchestrator finished")


if __name__ == "__main__":
    main()
