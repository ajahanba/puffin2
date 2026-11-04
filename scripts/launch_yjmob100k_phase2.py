#!/usr/bin/env python
"""YJMob100K phase 2 ordering, per the PI's instruction (2026-11-04).

After yjmob100k_plain_lp0.1 finishes, run ed_soft lambda4=1.0 next (not
the remaining plain lambda3 grid in order). Then run only the plain
lambda3 points needed to bracket ed_soft lambda4=1.0's utility against
the already-known plain_lp0.1 point, searching the remaining grid
{0.2, 0.3, 0.5, 0.8} in increasing lambda3 (utility decreases as
lambda3 grows, same monotonicity as every other dataset's sweep this
project has run) until one point's utility crosses to the other side
of the target -- never running the whole remaining grid, never
extrapolating. Then ed_soft lambda4=0.3, then ed_soft lambda4=3.0.
Any plain lambda3 points not needed for the bracket are queued last,
run only if time allows.

Reuses launch_yjmob100k.py's job runner (plateau audit inline, same
protocol) so corrected/audit-flagged seeds are handled identically.
"""
import json
import pathlib
import sys

sys.path.insert(0, ".")
import scripts.launch_yjmob100k as base

REMAINING_PLAIN_LP = [0.2, 0.3, 0.5, 0.8]
ED_SOFT_LAMBDA_PRIV = 0.3


def summary_utility(out_name):
    out_dir = base.OUT_BASE / out_name
    corrected = base.OUT_BASE / f"{out_name}_corrected"
    path = corrected / "summary.json" if corrected.exists() else out_dir / "summary.json"
    s = json.loads(path.read_text())
    return s["utility_mean"], path


def main():
    base.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_parquet = base.REPO / "reports/figures/yjmob100k_sample/unified_sample.parquet"

    def log(msg):
        base.log(msg)

    log("phase 2 controller starting: ed_soft lambda4=1.0 next, then bracket plain lambda3")

    # Step 1: ed_soft lambda4=1.0
    base.run_job_with_audit("yjmob100k_ed_soft_lambda1.0", "ed_soft", 1.0, ED_SOFT_LAMBDA_PRIV, sample_parquet)
    target_utility, target_path = summary_utility("yjmob100k_ed_soft_lambda1.0")
    log(f"ed_soft lambda4=1.0 utility={target_utility:.4f} (from {target_path.name})")

    # Step 2: bracket search against the already-known plain grid
    lp01_utility, _ = summary_utility("yjmob100k_plain_lp0.1")
    lp002_utility, _ = summary_utility("yjmob100k_plain_lp0.02")
    log(f"known plain points: lp0.02 utility={lp002_utility:.4f}, lp0.1 utility={lp01_utility:.4f}, "
        f"target(ed_soft 1.0)={target_utility:.4f}")

    run_for_bracket = []
    if lp002_utility >= target_utility >= lp01_utility:
        log("target already bracketed by lp0.02/lp0.1 -- no new plain points needed for the bracket")
    elif target_utility > lp002_utility:
        log("UNRESOLVABLE on the low-lambda3 end: target utility exceeds lp0.02's utility; "
            "would need lambda3 < 0.02, not in the planned grid. Not extrapolating. "
            "Will still run remaining plain points later if time allows.")
    else:
        # target_utility < lp01_utility: search upward through the remaining grid
        prev_utility = lp01_utility
        for lp in REMAINING_PLAIN_LP:
            out_name = f"yjmob100k_plain_lp{lp}"
            base.run_job_with_audit(out_name, "plain", 0.0, lp, sample_parquet)
            run_for_bracket.append(lp)
            this_utility, _ = summary_utility(out_name)
            log(f"bracket search: lp{lp} utility={this_utility:.4f} (target={target_utility:.4f})")
            if this_utility <= target_utility <= prev_utility:
                log(f"bracketed: lp{lp} utility={this_utility:.4f} <= target={target_utility:.4f} "
                    f"<= previous utility={prev_utility:.4f} -- stopping bracket search")
                break
            prev_utility = this_utility
        else:
            log("exhausted remaining plain grid without bracketing target -- "
                "UNRESOLVABLE on the high-lambda3 end with the current grid")

    # Step 3: ed_soft lambda4=0.3, then lambda4=3.0
    base.run_job_with_audit("yjmob100k_ed_soft_lambda0.3", "ed_soft", 0.3, ED_SOFT_LAMBDA_PRIV, sample_parquet)
    base.run_job_with_audit("yjmob100k_ed_soft_lambda3.0", "ed_soft", 3.0, ED_SOFT_LAMBDA_PRIV, sample_parquet)

    # Step 4: remaining plain lambda3 points not used for the bracket, lowest priority
    leftover = [lp for lp in REMAINING_PLAIN_LP if lp not in run_for_bracket]
    if leftover:
        log(f"running leftover plain lambda3 points if time allows: {leftover}")
        for lp in leftover:
            base.run_job_with_audit(f"yjmob100k_plain_lp{lp}", "plain", 0.0, lp, sample_parquet)

    log("YJMob100K phase 2 orchestrator finished")


if __name__ == "__main__":
    main()
