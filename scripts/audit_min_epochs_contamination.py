#!/usr/bin/env python
"""Audits every run directory flagged protocol_ok=False by
backfill_protocol_tags.py (missing min_epochs, i.e. predates the
min_epochs burn-in fix) for the actual cold-start-plateau signature:
a seed whose actual_epochs is suspiciously close to `patience` (stopped
right after the plateau's non-improving streak) AND whose utility sits
at or near the known chance-level plateau value for that run. This does
NOT assume the fix was needed -- it checks whether the bug actually fired,
seed by seed, so stale-but-unaffected runs aren't rerun needlessly, and
anything that IS affected is flagged for mandatory rerun before use in
any table.
"""
import json
import pathlib

BASE = pathlib.Path("/data/mashhadi/puffin2/reports/figures/retraining_to_convergence")

PLATEAU_EPOCH_MAX = 12  # stopped at or before this many epochs -- suspicious given patience=8
UTILITY_RATIO_THRESHOLD = 1.5  # utility within 50% of chance_level -- suspicious


def main():
    suspicious = []
    checked = 0
    for d in sorted(BASE.iterdir()):
        tag_path = d / "protocol_tag.json"
        summ_path = d / "summary.json"
        if not tag_path.exists() or not summ_path.exists():
            continue
        tag = json.load(open(tag_path))
        if tag.get("protocol_ok"):
            continue
        summ = json.load(open(summ_path))
        checked += 1
        epochs = summ.get("actual_epochs_per_encoder_seed", [])
        utils = summ.get("utility_per_encoder_seed", [])
        chance = summ.get("chance_level", 0)
        for seed_idx, (ep, u) in enumerate(zip(epochs, utils)):
            if ep <= PLATEAU_EPOCH_MAX and chance > 0 and u <= chance * (1 + UTILITY_RATIO_THRESHOLD):
                suspicious.append(dict(run=d.name, seed=seed_idx, epochs=ep, utility=u, chance_level=chance))

    print(f"Checked {checked} protocol_ok=False run directories.")
    if suspicious:
        print(f"\n{len(suspicious)} SUSPICIOUS seed-trainings found (plateau signature):")
        for s in suspicious:
            print(f"  {s['run']} seed={s['seed']}: epochs={s['epochs']} utility={s['utility']:.4f} chance={s['chance_level']:.4f}")
    else:
        print("\nNo suspicious seed-trainings found: none of the flagged runs show the "
              "cold-start-plateau signature (early stop at <=12 epochs with near-chance utility). "
              "These runs lack the min_epochs AUDIT TRAIL but show no sign the bug actually fired.")

    out = dict(checked=checked, n_suspicious=len(suspicious), suspicious=suspicious)
    json.dump(out, open(BASE.parent / "retraining_to_convergence_analysis" / "min_epochs_contamination_audit.json", "w"), indent=2)


if __name__ == "__main__":
    main()
