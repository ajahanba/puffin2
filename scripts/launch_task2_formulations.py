#!/usr/bin/env python
"""Task 2 of the fairness-reframing instruction (2026-11-02): principled
formulations, same tagged protocol. Waits for the GeoLife lambda3 sweep
and the 10-seed curve upgrade (both using GPU0/1) to finish first, per
the PI's explicit GPU priority order, then launches:

1. lipschitz_soft, similarity-basis=entropy, 3 datasets, lambda4 in
   {0.3,1.0,3.0}, 3 seeds (10 on GeoLife) -- upgrades the existing
   1-seed entropy-basis lipschitz runs to a real seed count.
2. lipschitz_soft, similarity-basis=location_distribution, same grid --
   the new variant.
3. ed_soft (PUFFIN), similarity-basis=location_distribution, GeoLife
   only, lambda4 in {0.3,1.0}, 10 seeds -- the minimum PUFFIN-with-a-
   different-similarity requirement.

Fault-tolerant: no set -e equivalent, each job's failure is logged and
the queue continues.
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


def log(msg):
    ts = datetime.datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def wait_for_gpu_queue():
    sweep_log = REPO / "logs/retraining_to_convergence/lambda3_sweep_controller.log"
    upgrade_log = REPO / "logs/retraining_to_convergence/upgrade_to_10seed.log"
    log("waiting for lambda3 sweep and 10-seed upgrade to finish (GPU priority order)...")
    while True:
        sweep_done = sweep_log.exists() and "sweep controller finished" in sweep_log.read_text()
        upgrade_done = upgrade_log.exists() and "upgrade-to-10seed batch finished" in upgrade_log.read_text()
        if sweep_done and upgrade_done:
            log("both finished -- proceeding with Task 2")
            return
        time.sleep(120)


def launch_job(label, cmd, gpu):
    log(f"LAUNCH {label} gpu={gpu}")
    job_log = REPO / f"logs/task2_formulations/{label}.log"
    logf = open(job_log, "w")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf


def wait_job(label, proc, logf):
    rc = proc.wait()
    logf.close()
    if rc != 0:
        log(f"FAILED {label} exit={rc} -- see logs/task2_formulations/{label}.log")
    else:
        log(f"OK {label}")
    return rc


def build_cmd(dataset, mechanism, lambda4, similarity_basis, n_seeds, out_name, gpu):
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
        "--device", f"cuda:{gpu}",
    ]
    cmd += ["--respect-traj-boundary"] if TRAJ_BOUNDARY[dataset] else ["--no-respect-traj-boundary"]
    return cmd


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    wait_for_gpu_queue()

    jobs = []
    for basis in ("entropy", "location_distribution"):
        for ds in ("geolife", "foursquare", "gowalla_sf_bay"):
            for l4 in (0.3, 1.0, 3.0):
                out_name = f"{ds}_lipschitz_{basis}_lambda{l4}_{N_SEEDS[ds]}seed"
                jobs.append((out_name, ds, "lipschitz", l4, basis, N_SEEDS[ds]))
    for l4 in (0.3, 1.0):
        out_name = f"geolife_ed_soft_location_distribution_lambda{l4}_10seed"
        jobs.append((out_name, "geolife", "ed_soft", l4, "location_distribution", 10))

    log(f"{len(jobs)} Task 2 jobs queued")
    pending = [j for j in jobs if not (OUT_BASE / j[0] / "summary.json").exists()]
    for j in jobs:
        if j not in pending:
            log(f"SKIP (already done): {j[0]}")
    i = 0
    while i < len(pending):
        batch = pending[i:i + 2]
        running = []
        for slot, (out_name, ds, mech, l4, basis, n_seeds) in enumerate(batch):
            cmd = build_cmd(ds, mech, l4, basis, n_seeds, out_name, slot)
            running.append((out_name, *launch_job(out_name, cmd, slot)))
        for out_name, proc, logf in running:
            wait_job(out_name, proc, logf)
        i += 2
    log("Task 2 orchestrator finished")


if __name__ == "__main__":
    main()
