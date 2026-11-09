#!/usr/bin/env python
"""Item 5 (PI instruction, 2026-11-09): trade-off figure, 2x2 panel
(one panel per dataset: GeoLife, Foursquare, Gowalla-SF-Bay; 4th panel
reserved/empty), x=Acc@1 (utility), y=decile-0 risk. Mo-PAE lambda3
curve (line, labeled points), no-privacy, PUFFIN lambda4 points,
Lipschitz (both similarity bases), Group-DRO (GeoLife only), GI-DP
epsilon points. TrajGAN omitted (not yet available). Seed-level error
bars (+-1 SEM across encoder seeds). Vector PDF, single-column width
(3.4in), colorblind-safe (Okabe-Ito palette), distinct markers per
mechanism family (not relying on color alone).
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

RETRAIN_BASE = pathlib.Path("reports/figures/retraining_to_convergence")
TASK2_BASE = pathlib.Path("reports/figures/task2_formulations")
GI_DP_BASE = pathlib.Path("reports/figures/gi_dp")
MIN_WINDOWS = 10

# Okabe-Ito colorblind-safe palette
COLOR = dict(
    mopae="#000000", no_privacy="#999999", puffin="#E69F00",
    lipschitz_entropy="#56B4E9", lipschitz_location="#009E73",
    group_dro="#CC79A7", gi_dp="#D55E00",
)
MARKER = dict(mopae="o", no_privacy="*", puffin="s", lipschitz_entropy="^",
              lipschitz_location="v", group_dro="D", gi_dp="P")


def decile0_seed_values(run_dir):
    summ = json.load(open(run_dir / "summary.json"))
    df = pd.read_csv(run_dir / "per_user_04d.csv")
    df = df[df.n_windows >= MIN_WINDOWS]
    seed_cols = sorted([c for c in df.columns if c.startswith("post_top1_encseed")],
                        key=lambda c: int(c.replace("post_top1_encseed", "")))
    util = np.array(summ["utility_per_encoder_seed"])
    d0 = np.array([df.loc[df.entropy_decile == 0, c].mean() for c in seed_cols])
    return util, d0


def sem(x):
    x = np.asarray(x)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def plot_point(ax, run_dir, label, color, marker, annotate=None):
    if not (run_dir / "summary.json").exists():
        return None
    util, d0 = decile0_seed_values(run_dir)
    ax.errorbar(util.mean(), d0.mean(), xerr=sem(util), yerr=sem(d0),
                fmt=marker, color=color, markersize=4, capsize=2, linewidth=0.8,
                label=label, zorder=3)
    if annotate:
        ax.annotate(annotate, (util.mean(), d0.mean()), fontsize=5, color=color,
                    xytext=(3, 3), textcoords="offset points")
    return util.mean(), d0.mean()


def plot_curve(ax, points, color, label, lp_labels):
    xs, ys = [], []
    for run_dir, lp in zip(points, lp_labels):
        if not (run_dir / "summary.json").exists():
            continue
        util, d0 = decile0_seed_values(run_dir)
        xs.append(util.mean())
        ys.append(d0.mean())
        ax.annotate(f"{lp}", (util.mean(), d0.mean()), fontsize=4.5, color=color,
                    xytext=(2, -5), textcoords="offset points")
    order = np.argsort(xs)
    xs, ys = np.array(xs)[order], np.array(ys)[order]
    ax.plot(xs, ys, "-", color=color, linewidth=1.0, marker="o", markersize=3, label=label, zorder=2)


DATASETS = ["geolife", "foursquare", "gowalla_sf_bay"]
LAMBDA3_STD = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8]
LAMBDA3_GEOLIFE = [0.02, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3]


def geolife_curve_dirs():
    out, labels = [], []
    for lp in LAMBDA3_GEOLIFE:
        if lp == 0.3:
            name = "geolife_plain_lambda0.0_10seed"
        elif lp == 2:
            name = "geolife_plain_lp2_10seed_corrected"
        elif lp == 3:
            corrected = RETRAIN_BASE / "geolife_plain_lp3_10seed_corrected"
            name = "geolife_plain_lp3_10seed_corrected" if corrected.exists() else "geolife_plain_lp3_10seed"
        elif lp == 1.2:
            name = "geolife_plain_lp1.2_10seed"
        else:
            name = f"geolife_plain_lp{lp}"
        out.append(RETRAIN_BASE / name)
        labels.append(lp)
    return out, labels


def std_curve_dirs(ds):
    out, labels = [], []
    for lp in LAMBDA3_STD:
        out.append(RETRAIN_BASE / f"{ds}_plain_lp{lp}")
        labels.append(lp)
    return out, labels


fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.4), sharex=False, sharey=False)
axes = axes.flatten()

for ax, ds in zip(axes[:3], DATASETS):
    if ds == "geolife":
        curve_dirs, curve_labels = geolife_curve_dirs()
    else:
        curve_dirs, curve_labels = std_curve_dirs(ds)
    plot_curve(ax, curve_dirs, COLOR["mopae"], "Mo-PAE $\\lambda_3$", curve_labels)

    plot_point(ax, RETRAIN_BASE / f"{ds}_no_privacy", "no-privacy", COLOR["no_privacy"], MARKER["no_privacy"])

    for l4 in (0.3, 1.0, 3.0):
        plot_point(ax, RETRAIN_BASE / f"{ds}_ed_soft_lambda{l4}"
                   if ds != "geolife" else RETRAIN_BASE / f"geolife_ed_soft_lambda{l4}_10seed",
                   "PUFFIN" if l4 == 0.3 else None, COLOR["puffin"], MARKER["puffin"], annotate=f"{l4}")

    n_seed_tag = "10seed" if ds == "geolife" else "3seed"
    for l4 in (0.3, 1.0, 3.0):
        plot_point(ax, TASK2_BASE / f"{ds}_lipschitz_entropy_lambda{l4}_{n_seed_tag}",
                   "Lipschitz (entropy)" if l4 == 0.3 else None,
                   COLOR["lipschitz_entropy"], MARKER["lipschitz_entropy"], annotate=f"{l4}")
        plot_point(ax, TASK2_BASE / f"{ds}_lipschitz_location_distribution_lambda{l4}_{n_seed_tag}",
                   "Lipschitz (location)" if l4 == 0.3 else None,
                   COLOR["lipschitz_location"], MARKER["lipschitz_location"], annotate=f"{l4}")

    if ds == "geolife":
        for l4 in (0.3, 1.0):
            plot_point(ax, TASK2_BASE / f"geolife_group_dro_lambda{l4}_3seed",
                       "Group-DRO" if l4 == 0.3 else None, COLOR["group_dro"], MARKER["group_dro"], annotate=f"{l4}")

    gi_dp_dirs, gi_dp_labels = [], []
    for eps in (0.0005, 0.001, 0.002, 0.005, 0.01):
        gi_dp_dirs.append(GI_DP_BASE / f"{ds}_gi_dp_eps{eps}")
        gi_dp_labels.append(eps)
    plot_curve(ax, gi_dp_dirs, COLOR["gi_dp"], "GI-DP $\\epsilon$", gi_dp_labels)

    ax.set_title(ds.replace("_", "-"), fontsize=7)
    ax.set_xlabel("Acc@1 (utility)", fontsize=6)
    ax.set_ylabel("decile-0 risk", fontsize=6)
    ax.tick_params(labelsize=5)

axes[3].axis("off")
handles, labels_ = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels_, handles))
axes[3].legend(by_label.values(), by_label.keys(), loc="center", fontsize=5.5, frameon=False)

plt.tight_layout()
out_pdf = "reports/figures/paper/tradeoff_figure.pdf"
plt.savefig(out_pdf)
plt.savefig("reports/figures/paper/tradeoff_figure.png", dpi=200)
print(f"wrote {out_pdf}")
