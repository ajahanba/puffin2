#!/usr/bin/env python
"""Generates Figure 1 (raw vs. post-Mo-PAE risk by decile, per dataset)
and Figure 2 (utility vs. decile-0 risk fronts, plain-Mo-PAE reference
curve and no-privacy point, per dataset) for the paper. Run from the
repo root: `python paper/scripts/generate_plots.py`. Writes PDF figures
to paper/figures/, referenced from the paper via \\includegraphics.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["geolife", "foursquare", "gowalla_nyc"]
LABELS = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
MIN_WINDOWS = 10


def fig1_raw_vs_post_by_decile():
    d = json.load(open(ROOT / "reports/figures/paper_section8/geolife_foursquare_disparate_risk.json"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
    deciles = list(range(10))
    for ax, ds in zip(axes, DATASETS):
        red = d[ds]["reduction"]
        raw = [red[str(dec)]["raw_mean"] if str(dec) in red else red[dec]["raw_mean"] for dec in deciles]
        post = [red[str(dec)]["post_mean"] if str(dec) in red else red[dec]["post_mean"] for dec in deciles]
        ax.plot(deciles, raw, marker="o", color="#555555", label="raw (pre-mechanism)", linewidth=1.5)
        ax.plot(deciles, post, marker="s", color="#c1121f", label="post (plain Mo-PAE)", linewidth=1.5)
        ax.set_title(LABELS[ds], fontsize=10)
        ax.set_xlabel("entropy decile (0=lowest)", fontsize=8)
        ax.set_xticks(deciles)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("re-identification risk (top-1)", fontsize=8)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGDIR / "fig1_raw_vs_post_by_decile.pdf", bbox_inches="tight")
    print(f"wrote {FIGDIR / 'fig1_raw_vs_post_by_decile.pdf'}")


def _load_summary_and_df(path):
    summ = json.load(open(path / "summary.json"))
    df = pd.read_csv(path / "per_user_04d.csv")
    df = df[df["n_windows"] >= MIN_WINDOWS].copy()
    return summ, df


def _decile0_risk(df):
    sub = df[df.entropy_decile == 0]
    return float(sub["post_top1"].mean()) if len(sub) else None


def fig2_utility_vs_risk_fronts():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        # plain-Mo-PAE reference curve, 40-epoch points only
        curve_pts = []
        for lp in (0.02, 0.1, 0.2, 0.5, 0.8):
            p = ROOT / f"reports/figures/phase4e_refcurve/{ds}_plain_lp{lp}"
            if (p / "summary.json").exists():
                summ, df = _load_summary_and_df(p)
                curve_pts.append((summ["utility_mean"], _decile0_risk(df)))
        curve_pts.sort(key=lambda t: t[0])
        if curve_pts:
            xs, ys = zip(*curve_pts)
            ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.5, markersize=4,
                     label="plain Mo-PAE ($\\lambda_3$ swept)")

        # no-privacy ceiling point (40 epochs)
        np_path = ROOT / f"reports/figures/phase4e_refit40/{ds}_no_privacy_40ep"
        if (np_path / "summary.json").exists():
            summ, df = _load_summary_and_df(np_path)
            ax.scatter([summ["utility_mean"]], [_decile0_risk(df)], marker="*", s=90,
                       color="black", zorder=5, label="no-privacy reference")

        # formulation points (ed_soft, floor) from the original 20-epoch batch
        mech_markers = {"ed_soft": ("^", "#2ca02c"), "floor": ("D", "#ff7f0e")}
        for mech, (marker, color) in mech_markers.items():
            lambdas = (0.3, 1.0, 3.0) if (mech == "ed_soft" or ds == "geolife") else ()
            xs, ys = [], []
            for l4 in lambdas:
                p = ROOT / f"reports/figures/phase4d/{ds}_{mech}_lambda{l4}"
                if (p / "summary.json").exists():
                    summ, df = _load_summary_and_df(p)
                    xs.append(summ["utility_mean"])
                    ys.append(_decile0_risk(df))
            if xs:
                ax.scatter(xs, ys, marker=marker, s=40, color=color, label=mech, zorder=4)

        ax.set_title(LABELS[ds], fontsize=10)
        ax.set_xlabel("utility (next-location top-1)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("decile-0 risk (top-1)", fontsize=8)
    # GeoLife's axis (index 0) has all four series (curve, no-privacy, ed_soft, floor)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(FIGDIR / "fig2_utility_vs_risk_fronts.pdf", bbox_inches="tight")
    print(f"wrote {FIGDIR / 'fig2_utility_vs_risk_fronts.pdf'}")


if __name__ == "__main__":
    fig1_raw_vs_post_by_decile()
    fig2_utility_vs_risk_fronts()
