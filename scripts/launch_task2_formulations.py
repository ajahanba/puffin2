#!/usr/bin/env python
"""Task 2 of the fairness-reframing instruction (2026-11-02): principled
formulations, same tagged protocol. Per the PI's GPU-reallocation
instruction: GPU0 stays on the lambda3 sweep; this runs on GPU1 alone,
starting immediately (not waiting for the 10-seed curve upgrade, which
now runs AFTER Task 2 -- it only tightens CIs, Task 2 is the core
result). Sequential (one GPU), in this exact order:
  1. GeoLife: ed_soft_location (lambda4 in {0.3,1.0}, 10 seeds)
  2. GeoLife: lipschitz, similarity-basis=entropy (lambda4 in {0.3,1.0,3.0}, 10 seeds)
  3. GeoLife: lipschitz, similarity-basis=location_distribution (same grid)
  4. Gowalla-SF-Bay: lipschitz, both bases (lambda4 in {0.3,1.0,3.0}, 3 seeds)
  5. Foursquare: lipschitz, both bases (lambda4 in {0.3,1.0,3.0}, 3 seeds)

Fault-tolerant: each job's failure is logged and the queue continues.
Reports a running ETA for the remaining queue once enough jobs have
timed in.
"""
import datetime
import json
import pathlib
import subprocess
import time

REPO = pathlib.Path("/data/mashhadi/puffin2")
OUT_BASE = REPO / "reports/figures/task2_formulations"
LOG_PATH = REPO / "logs/task2_formulations/orchestrator.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

DATASETS = {"geolife": "data/processed/geolife/unified.parquet",
            "foursquare": "data/processed/foursquare/unified.parquet",
            "gowalla_sf_bay": "data/processed/gowalla_sf_bay/unified.parquet"}
N_SEEDS = {"geolife": 10, "foursquare": 3, "gowalla_sf_bay": 3}
TRAJ_BOUNDARY = {"geolife": True, "foursquare": False, "gowalla_sf_bay": False}
GPU = 1


def log(msg):
    ts = datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def build_cmd(dataset, mechanism, lambda4, similarity_basis, n_seeds, out_name):
    seeds = ",".join(str(s) for s in range(n_seeds))
    cmd = [
        str(REPO / ".venv/bin/python"), "scripts/run_phase4d_approved_batch.py",
        "--dataset", dataset, "--parquet", DATASETS[dataset],
        "--out-dir", str(OUT_BASE / out_name), "--mechanism", mechanism,
        "--lambda4", str(lambda4), "--lambda-priv", "0.3",
        "--similarity-basis", similarity_basis,
        "--sl", "10", "--max-windows-per-user", "200",
        "--mopae-epochs", "80", "--patience", "8", "--min-epochs", "20", "--val-frac", "0.1",
        "--attack-epochs", "25", "--encoder-seeds", seeds, "--attacker-seeds", "0,1,2,3,4",
        "--device", f"cuda:{GPU}",
    ]
    cmd += ["--respect-traj-boundary"] if TRAJ_BOUNDARY[dataset] else ["--no-respect-traj-boundary"]
    return cmd


def run_job(out_name, cmd, job_times):
    if (OUT_BASE / out_name / "summary.json").exists():
        log(f"SKIP (already done): {out_name}")
        return
    log(f"LAUNCH {out_name} gpu={GPU}")
    t0 = time.time()
    job_log = REPO / f"logs/task2_formulations/{out_name}.log"
    with open(job_log, "w") as logf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0
    job_times.append(elapsed)
    if rc != 0:
        log(f"FAILED {out_name} exit={rc} ({elapsed/60:.1f} min) -- see {job_log}")
    else:
        log(f"OK {out_name} ({elapsed/60:.1f} min)")


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    log("Task 2 starting on GPU1 (GPU0 stays on the lambda3 sweep), waiting for nothing")

    jobs = []
    # 1. GeoLife ed_soft_location
    for l4 in (0.3, 1.0):
        jobs.append((f"geolife_ed_soft_location_distribution_lambda{l4}_10seed",
                      "geolife", "ed_soft", l4, "location_distribution", 10))
    # 2+3. GeoLife lipschitz, both bases
    for basis in ("entropy", "location_distribution"):
        for l4 in (0.3, 1.0, 3.0):
            jobs.append((f"geolife_lipschitz_{basis}_lambda{l4}_10seed",
                          "geolife", "lipschitz", l4, basis, 10))
    # 4. Gowalla-SF-Bay lipschitz, both bases
    for basis in ("entropy", "location_distribution"):
        for l4 in (0.3, 1.0, 3.0):
            jobs.append((f"gowalla_sf_bay_lipschitz_{basis}_lambda{l4}_3seed",
                          "gowalla_sf_bay", "lipschitz", l4, basis, 3))
    # 5. Foursquare lipschitz, both bases
    for basis in ("entropy", "location_distribution"):
        for l4 in (0.3, 1.0, 3.0):
            jobs.append((f"foursquare_lipschitz_{basis}_lambda{l4}_3seed",
                          "foursquare", "lipschitz", l4, basis, 3))

    log(f"{len(jobs)} Task 2 jobs queued, in order: " + ", ".join(j[0] for j in jobs))
    job_times = []
    for i, (out_name, ds, mech, l4, basis, n_seeds) in enumerate(jobs):
        cmd = build_cmd(ds, mech, l4, basis, n_seeds, out_name)
        run_job(out_name, cmd, job_times)
        if len(job_times) == 3:
            avg = sum(job_times) / len(job_times)
            remaining = len(jobs) - (i + 1)
            eta_hours = avg * remaining / 3600
            log(f"ETA (based on first 3 real runs, avg {avg/60:.1f} min/job): "
                f"~{eta_hours:.1f} h remaining for {remaining} jobs")
    log("Task 2 orchestrator finished")


if __name__ == "__main__":
    main()
