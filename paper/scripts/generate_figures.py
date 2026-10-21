#!/usr/bin/env python
"""Generates every LaTeX table/figure the paper uses, from the reports'
already-committed JSON data -- never hand-edited. Run from the repo
root: `python paper/scripts/generate_figures.py`.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLES = pathlib.Path(__file__).resolve().parents[1] / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def write(name, content):
    (TABLES / name).write_text(content)
    print(f"wrote {TABLES / name}")


def gen_gowalla_disparate_risk_table():
    d = json.load(open(ROOT / "reports/figures/phase2d/decile_reduction_metrics.json"))
    names = {"gowalla_nyc": "Gowalla-NYC", "gowalla_la": "Gowalla-LA",
             "gowalla_chicago": "Gowalla-Chicago", "gowalla_sf_bay": "Gowalla-SF-Bay"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Residual-risk disparity (post-mechanism risk, decile 0 $\div$ decile 9) "
        r"and relative reduction at deciles 0 and 9, plain Mo-PAE, all four Gowalla metro "
        r"subsets. Bootstrap 95\% CIs in brackets. Generated from "
        r"\texttt{reports/figures/phase2d/decile\_reduction\_metrics.json}.}",
        r"\label{tab:gowalla-disparate-risk}",
        r"\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Dataset & Rel.\ reduction, decile 0 & Rel.\ reduction, decile 9 & Residual disparity (0/9) \\",
        r"\midrule",
    ]
    for ds, label in names.items():
        red = d[ds]["decile_reduction"]
        disp = d[ds]["residual_risk_disparity"]
        r0 = red["0"]["relative_reduction"]
        r9 = red["9"]["relative_reduction"]
        lines.append(
            f"{label} & {r0['mean']:.3f} [{r0['ci_lo']:.3f}, {r0['ci_hi']:.3f}] & "
            f"{r9['mean']:.3f} [{r9['ci_lo']:.3f}, {r9['ci_hi']:.3f}] & "
            f"{disp['point_estimate']:.2f}$\\times$ [{disp['ci_lo']:.2f}, {disp['ci_hi']:.2f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("gowalla_disparate_risk.tex", "\n".join(lines) + "\n")


def gen_adaptive_gap_decomposition_table():
    results = {}
    for ds in ["geolife", "foursquare", "gowalla_nyc"]:
        results[ds] = json.load(open(ROOT / f"reports/figures/phase3d/{ds}/summary.json"))
    names = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Gap decomposition (Checkpoint 3d): mean of the "
        r"side-information, specialization, and realistic gap components "
        r"across all deciles, per dataset, with the count of (dataset, "
        r"decile) cells whose bootstrap CI excludes zero. Generated from "
        r"\texttt{reports/figures/phase3d/*/summary.json}.}",
        r"\label{tab:gap-decomposition}",
        r"\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Dataset & Side info.\ (b$-$a) & Specialization (c$-$b) & Realistic (e$-$a) \\",
        r"\midrule",
    ]
    for ds, label in names.items():
        gaps = results[ds]["gap_components_by_decile"]
        si = gaps["side_information_b_minus_a"]
        sp = gaps["specialization_c_minus_b"]
        re = gaps["realistic_e_minus_a"]

        def summarize(comp):
            vals = [v["mean"] for v in comp.values()]
            n_excl = sum(1 for v in comp.values() if v["ci_excludes_zero"])
            return f"{sum(vals)/len(vals):+.3f} ({n_excl}/{len(vals)} CI excl.\\ 0)"

        lines.append(f"{label} & {summarize(si)} & {summarize(sp)} & {summarize(re)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("gap_decomposition.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    gen_gowalla_disparate_risk_table()
    gen_adaptive_gap_decomposition_table()
