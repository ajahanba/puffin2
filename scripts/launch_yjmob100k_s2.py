#!/usr/bin/env python
"""YJMob100K training, SECOND independent 2000-user sample (seed=1),
same filters/user-ID remap/split rules as the first sample
(reports/figures/yjmob100k_sample/) via scripts/sample_yjmob100k.py
--seed 1 --out-dir reports/figures/yjmob100k_sample_s2. Same protocol
as the first sample (current run_phase4d_approved_batch.py, standard
external attacker, inline plateau audit), restricted to the 3
configurations needed to check whether PUFFIN ed_soft lambda4=1.0's
utility is bracketed by plain Mo-PAE lambda3 in {0.1, 0.2} -- the same
pair that bracketed it in the first sample. 3 encoder seeds each.

Run twice, in parallel, with disjoint --jobs so the two GPUs don't race
on building the shared sample parquet (--prepare-only builds it once,
synchronously, before either GPU job list starts):
    python scripts/launch_yjmob100k_s2.py --prepare-only
    python scripts/launch_yjmob100k_s2.py --gpu 0 --jobs plain_lp0.1,plain_lp0.2
    python scripts/launch_yjmob100k_s2.py --gpu 1 --jobs ed_soft_lambda1.0
"""
import argparse
import datetime
import json
import pathlib
import subprocess

import sys
sys.path.insert(0, ".")
from puffin.eval.plateau_audit import audit_run, apply_audit_corrections

REPO = pathlib.Path("/data/mashhadi/puffin2")
DATASET = "yjmob100k"
OUT_BASE = REPO / "reports/figures/retraining_to_convergence"
LOG_DIR = REPO / "logs/retraining_to_convergence"
N_SEEDS = 3
SAMPLE_DIR = REPO / "reports/figures/yjmob100k_sample_s2"
SAMPLE_MANIFEST = SAMPLE_DIR / "sample_manifest.json"
SAMPLE_PARQUET = SAMPLE_DIR / "unified_sample.parquet"
FULL_PARQUET = REPO / "data/processed/yjmob100k_full/unified.parquet"

JOBS = {
    "plain_lp0.1": ("yjmob100k_s2_plain_lp0.1", "plain", 0.0, 0.1),
    "plain_lp0.2": ("yjmob100k_s2_plain_lp0.2", "plain", 0.0, 0.2),
    "ed_soft_lambda1.0": ("yjmob100k_s2_ed_soft_lambda1.0", "ed_soft", 1.0, 0.3),
}


def log(msg, gpu_tag):
    ts = datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y")
    line = f"{ts} [s2 gpu{gpu_tag}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "yjmob100k_s2_orchestrator.log", "a") as f:
        f.write(line + "\n")


def sample_users():
    return json.loads(SAMPLE_MANIFEST.read_text())["user_ids"]


def filter_to_sample(parquet_path, user_ids, out_path):
    """Same remap logic as the first sample's launch_yjmob100k.py --
    dense 1..N user_id remap, built once, persisted alongside the
    filtered parquet so every run's user_id numbering agrees."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    sub = df[df["user_id"].isin(user_ids)].copy()
    sorted_ids = sorted(user_ids)
    remap = {old_id: new_id for new_id, old_id in enumerate(sorted_ids, start=1)}
    sub["user_id"] = sub["user_id"].map(remap)
    sub.to_parquet(out_path, index=False)
    (out_path.parent / "user_id_remap.json").write_text(
        json.dumps({str(k): v for k, v in remap.items()}, indent=2))
    return out_path


def build_cmd(mechanism, lambda4, lambda_priv, out_name, seeds, gpu):
    seed_str = ",".join(str(s) for s in seeds)
    return [
        str(REPO / ".venv/bin/python"), "scripts/run_phase4d_approved_batch.py",
        "--dataset", DATASET, "--parquet", str(SAMPLE_PARQUET),
        "--out-dir", str(OUT_BASE / out_name), "--mechanism", mechanism,
        "--lambda4", str(lambda4), "--lambda-priv", str(lambda_priv),
        "--sl", "10", "--max-windows-per-user", "200",
        "--mopae-epochs", "80", "--patience", "8", "--min-epochs", "20", "--val-frac", "0.1",
        "--attack-epochs", "25", "--encoder-seeds", seed_str, "--attacker-seeds", "0,1,2,3,4",
        "--device", f"cuda:{gpu}", "--no-respect-traj-boundary",
    ]


def run_job_with_audit(out_name, mechanism, lambda4, lambda_priv, gpu):
    out_dir = OUT_BASE / out_name
    if (out_dir / "summary.json").exists():
        log(f"SKIP (already done): {out_name}", gpu)
        return
    cmd = build_cmd(mechanism, lambda4, lambda_priv, out_name, range(N_SEEDS), gpu)
    log(f"LAUNCH {out_name} gpu={gpu}", gpu)
    job_log = LOG_DIR / f"{out_name}.log"
    with open(job_log, "w") as logf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f"FAILED {out_name} exit={rc} -- see {job_log}", gpu)
        return
    log(f"OK {out_name}", gpu)

    audit = audit_run(out_dir, DATASET, str(SAMPLE_PARQUET), f"cuda:{gpu}", traj_boundary=False)
    if audit["n_flagged"] == 0:
        log(f"AUDIT {out_name}: 0 flagged seeds, clean", gpu)
        return
    log(f"AUDIT {out_name}: {audit['n_flagged']} flagged, {audit['n_genuine_collapse']} genuine collapse "
        f"-- naive_mean={audit['original_mean']:.4f} audited_mean={audit['audited_mean']:.4f}", gpu)
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
    log(f"wrote {corrected_dir.name}", gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--jobs", default="", help="comma-separated keys from JOBS")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    users = sample_users()
    if args.prepare_only:
        if SAMPLE_PARQUET.exists():
            print(f"{SAMPLE_PARQUET} already exists, nothing to prepare")
            return
        print(f"filtering full unified.parquet down to the {len(users)}-user sample (seed=1)...")
        filter_to_sample(FULL_PARQUET, users, SAMPLE_PARQUET)
        print(f"wrote {SAMPLE_PARQUET}")
        return

    assert SAMPLE_PARQUET.exists(), f"{SAMPLE_PARQUET} missing -- run --prepare-only first"
    job_keys = [k for k in args.jobs.split(",") if k]
    log(f"{len(job_keys)} jobs queued: {job_keys}", args.gpu)
    for key in job_keys:
        out_name, mech, l4, lp = JOBS[key]
        run_job_with_audit(out_name, mech, l4, lp, args.gpu)
    log("yjmob100k_s2 orchestrator finished", args.gpu)


if __name__ == "__main__":
    main()
