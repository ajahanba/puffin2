"""Phase 3.3 leveling-down check: ΔP(u) = P_mechanism(u) - P_baseline(u),
where both are privacy_gain series for the SAME users under two runs
matched on utility (baseline = Mo-PAE with the mechanism's own lambda1-3,
lambda4=0, tuned to the mechanism's Acc@5 +/- 0.5pp -- that utility-matching
search is an experiment-design step for whichever phase runs the actual
comparison, e.g. Phase 4/5; this module only computes the metric once both
gain series already exist at matched utility, and does not do the tuning
itself).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def leveling_down_check(gain_mechanism: pd.Series, gain_baseline: pd.Series) -> dict:
    """Both indexed by user_id (or any common key) so they align correctly
    -- caller's responsibility that both cover the same users. Reports the
    fraction harmed/helped/unchanged, and whether the harmed users tend to
    be the ones the baseline already protected well (if so, this mechanism
    is leveling down rather than lifting the worst-off)."""
    gain_mechanism, gain_baseline = gain_mechanism.align(gain_baseline, join="inner")
    delta = gain_mechanism - gain_baseline

    harmed = delta < -1e-9
    helped = delta > 1e-9
    unchanged = ~harmed & ~helped

    out = dict(
        n_users=int(len(delta)),
        frac_harmed=float(harmed.mean()), frac_helped=float(helped.mean()), frac_unchanged=float(unchanged.mean()),
        mean_delta=float(delta.mean()),
    )
    if harmed.sum() >= 2:
        harmed_baseline_gain = float(gain_baseline[harmed].mean())
        all_baseline_gain = float(gain_baseline.mean())
        out["harmed_users_baseline_gain_mean"] = harmed_baseline_gain
        out["all_users_baseline_gain_mean"] = all_baseline_gain
        out["leveling_down"] = harmed_baseline_gain > all_baseline_gain
    return out
