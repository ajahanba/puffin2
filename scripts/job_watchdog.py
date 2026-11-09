#!/usr/bin/env python
"""Generic per-job watchdog, PI instruction (2026-11-09): if a job's wall
time exceeds `--multiplier` (default 3x) its expected duration, capture
full diagnostics (nvidia-smi incl. all GPU processes, top -b -n1, free -g,
uptime, a py-spy stack dump of the job, and a process tree) to
reports/diagnostics/<timestamp>_<job-name>/, write a SUMMARY.txt, and
print a notification -- WITHOUT killing the job. Captures once per
watchdog process (re-run manually for a fresh capture on a job that stays
slow). Baseline is the watchdog's OWN start time, so it must be launched
right when the job starts (document this at the call site, not derived
from /proc to avoid a psutil dependency).

Usage: nohup .venv/bin/python scripts/job_watchdog.py --pid <PID> \
    --expected-seconds <N> --job-name <NAME> &
"""
import argparse
import datetime
import os
import pathlib
import subprocess
import time

REPO = pathlib.Path("/data/mashhadi/puffin2")
DIAG_BASE = REPO / "reports/diagnostics"


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_capture(cmd, out_dir, outfile, timeout=30):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        (out_dir / outfile).write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    except Exception as e:
        (out_dir / outfile).write_text(f"FAILED: {e}")


def capture_diagnostics(pid, job_name, elapsed, expected):
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = DIAG_BASE / f"{ts}_{job_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_capture("nvidia-smi", out_dir, "nvidia_smi.txt")
    run_capture("nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv",
                out_dir, "nvidia_smi_processes.txt")
    run_capture("top -b -n1", out_dir, "top.txt")
    run_capture("free -g", out_dir, "free.txt")
    run_capture("uptime", out_dir, "uptime.txt")
    run_capture(f"{REPO}/.venv/bin/py-spy dump --pid {pid}", out_dir, "py_spy_dump.txt", timeout=60)
    run_capture("ps -ef", out_dir, "ps_ef.txt")

    note = (f"WATCHDOG TRIGGERED: job={job_name} pid={pid} "
            f"elapsed={elapsed/3600:.2f}h expected={expected/3600:.2f}h "
            f"ratio={elapsed/expected:.1f}x\nDiagnostics saved to {out_dir}\n")
    (out_dir / "SUMMARY.txt").write_text(note)
    print(note, flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--expected-seconds", type=float, required=True)
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--poll-seconds", type=float, default=120)
    ap.add_argument("--multiplier", type=float, default=3.0)
    args = ap.parse_args()

    start = time.time()
    threshold = args.multiplier * args.expected_seconds
    triggered = False
    print(f"[watchdog {args.job_name}] watching pid {args.pid}, expected {args.expected_seconds:.0f}s, "
          f"trigger at {threshold:.0f}s ({args.multiplier}x)", flush=True)

    while True:
        if not pid_alive(args.pid):
            print(f"[watchdog {args.job_name}] pid {args.pid} no longer running, exiting", flush=True)
            break
        elapsed = time.time() - start
        if not triggered and elapsed > threshold:
            triggered = True
            capture_diagnostics(args.pid, args.job_name, elapsed, args.expected_seconds)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
