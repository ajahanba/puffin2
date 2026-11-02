#!/usr/bin/env python
"""Sequential, data-dependent GeoLife lambda3 (lambda_priv) sweep, per the
PI's explicit instruction (2026-11-02): extend the reference curve with
new points {1.2,2,3,5,8,12,20} beyond the existing {0.02,...,0.8}, in
ascending order, at 10 encoder seeds each, holding lambda1/lambda2/lambda4
and the early-stopping protocol fixed. After each new point finishes,
check its mean utility; once a point's utility is <=0.70, run exactly one
more point beyond it, then stop. If utility is still >0.70 at
lambda3=20, stop and report -- do NOT start the lambda2 fallback sweep
without explicit sign-off.
"""
import datetime
import json
import pathlib
import subprocess

REPO = pathlib.Path("/data/mashhadi/puffin2")
OUT_BASE = REPO / "reports/figures/retraining_to_convergence"
LOG_PATH = REPO / "logs/retraining_to_convergence/lambda3_sweep_controller.log"

NEW_POINTS = [1.2, 2, 3, 5, 8, 12, 20]
N_SEEDS = 10
UTILITY_STOP_THRESHOLD = 0.70
GPU = 0


def log(msg):
    ts = datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_point(lp):
    out_dir = OUT_BASE / f"geolife_plain_lp{lp}_10seed"
    if (out_dir / "summary.json").exists():
        log(f"SKIP (already done): lp={lp}")
        return json.load(open(out_dir / "summary.json"))
    cmd = [
        str(REPO / ".venv/bin/python"), "scripts/run_phase4d_approved_batch.py",
        "--dataset", "geolife", "--parquet", "data/processed/geolife/unified.parquet",
        "--out-dir", str(out_dir), "--mechanism", "plain", "--lambda4", "0.0",
        "--lambda-priv", str(lp), "--no-respect-traj-boundary", "--sl", "10",
        "--max-windows-per-user", "200", "--mopae-epochs", "80", "--patience", "8",
        "--min-epochs", "20", "--val-frac", "0.1", "--attack-epochs", "25",
        "--encoder-seeds", ",".join(str(s) for s in range(N_SEEDS)),
        "--attacker-seeds", "0,1,2,3,4", "--device", f"cuda:{GPU}",
    ]
    log(f"LAUNCH lp={lp} gpu={GPU}")
    job_log = REPO / f"logs/retraining_to_convergence/geolife_plain_lp{lp}_10seed.log"
    with open(job_log, "w") as logf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f"FAILED lp={lp} exit={rc} -- see {job_log}")
        return None
    summ = json.load(open(out_dir / "summary.json"))
    log(f"OK lp={lp} utility_mean={summ['utility_mean']:.4f} "
        f"epochs={summ['actual_epochs_per_encoder_seed']}")
    return summ


def main():
    log(f"sweep controller starting, new points {NEW_POINTS}, stop threshold utility<={UTILITY_STOP_THRESHOLD}")
    crossed = False
    for lp in NEW_POINTS:
        summ = run_point(lp)
        if summ is None:
            log(f"ABORTING sweep at lp={lp}: job failed, refusing to guess whether to continue")
            return
        util = summ["utility_mean"]
        if crossed:
            log(f"Ran exactly one point past the <=0.70 crossing (lp={lp}, utility={util:.4f}). "
                f"STOPPING per the stopping rule.")
            return
        if util <= UTILITY_STOP_THRESHOLD:
            log(f"CROSSED at lp={lp} (utility={util:.4f} <= {UTILITY_STOP_THRESHOLD}). "
                f"Running exactly one more point, then stopping.")
            crossed = True
    if not crossed:
        log(f"Reached lp=20 without utility dropping to <={UTILITY_STOP_THRESHOLD}. "
            f"STOPPING AND REPORTING. NOT starting the lambda2 fallback sweep -- needs explicit sign-off.")
    log("sweep controller finished")


if __name__ == "__main__":
    main()
