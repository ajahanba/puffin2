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


def gen_main_datasets_disparate_risk_table():
    d = json.load(open(ROOT / "reports/figures/paper_section8/geolife_foursquare_disparate_risk.json"))
    names = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Residual-risk disparity under plain Mo-PAE -- ratio (decile 0 $\div$ decile 9) "
        r"and absolute difference (decile 0 $-$ decile 9) of post-mechanism risk -- and relative "
        r"risk reduction at deciles 0 and 9, GeoLife, Foursquare, and Gowalla-NYC. Bootstrap 95\% "
        r"CIs in brackets. Generated from "
        r"\texttt{reports/figures/paper\_section8/geolife\_foursquare\_disparate\_risk.json}.}",
        r"\label{tab:main-disparate-risk}",
        r"\small",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Dataset & $n$ & Rel.\ reduction, decile 0 & Rel.\ reduction, decile 9 & "
        r"Disparity ratio (0/9) & Disparity difference (0$-$9) \\",
        r"\midrule",
    ]
    for ds, label in names.items():
        v = d[ds]
        red = v["reduction"]
        r0 = red["0"]["relative_reduction"]
        r9 = red["9"]["relative_reduction"]
        ratio = v["ratio"]
        diff = v["difference"]
        lines.append(
            f"{label} & {v['n_users']} & {r0['mean']:+.3f} [{r0['ci_lo']:+.3f}, {r0['ci_hi']:+.3f}] & "
            f"{r9['mean']:+.3f} [{r9['ci_lo']:+.3f}, {r9['ci_hi']:+.3f}] & "
            f"{ratio['point_estimate']:.2f}$\\times$ [{ratio['ci_lo']:.2f}, {ratio['ci_hi']:.2f}] & "
            f"{diff['point_estimate']:.3f} [{diff['ci_lo']:.3f}, {diff['ci_hi']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("main_datasets_disparate_risk.tex", "\n".join(lines) + "\n")


def gen_adaptive_gap_decomposition_table():
    results = {}
    for ds in ["geolife", "foursquare", "gowalla_nyc"]:
        results[ds] = json.load(open(ROOT / f"reports/figures/phase3d/{ds}/summary.json"))
    names = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Gap decomposition: mean of the "
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


def gen_04d_decile_reduction_table():
    names = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
    mech_labels = {"plain_lambda0.0": "plain Mo-PAE", "no_privacy_lambda0.0": "no-privacy ref.",
                   "ed_soft_lambda0.3": "ed\\_soft $\\lambda_4$=0.3", "ed_soft_lambda1.0": "ed\\_soft $\\lambda_4$=1.0",
                   "ed_soft_lambda3.0": "ed\\_soft $\\lambda_4$=3.0",
                   "floor_lambda0.3": "floor $\\lambda_4$=0.3", "floor_lambda1.0": "floor $\\lambda_4$=1.0",
                   "floor_lambda3.0": "floor $\\lambda_4$=3.0"}
    order = ["plain_lambda0.0", "no_privacy_lambda0.0", "ed_soft_lambda0.3", "ed_soft_lambda1.0",
             "ed_soft_lambda3.0", "floor_lambda0.3", "floor_lambda1.0", "floor_lambda3.0"]
    d = json.load(open(ROOT / "reports/figures/phase4d_analysis/decile_reduction_metrics.json"))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Utility, relative risk reduction at deciles 0 and 9, and "
        r"residual-risk disparity (RRD, post-mechanism risk decile 0 $\div$ decile 9), for "
        r"each fairness formulation and dataset. Bootstrap 95\% CIs in brackets; $\geq 3$ encoder seeds per point. "
        r"Generated from \texttt{reports/figures/phase4d\_analysis/decile\_reduction\_metrics.json}.}",
        r"\label{tab:04d-decile-reduction}",
        r"\small",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Dataset & Run & Utility & Rel.\ red., decile 0 & Rel.\ red., decile 9 & RRD (0/9) \\",
        r"\midrule",
    ]
    for ds, ds_label in names.items():
        for key in order:
            if key not in d[ds]:
                continue
            v = d[ds][key]
            red = v["reduction"]
            rrd = v["residual_risk_disparity"]
            r0, r9 = red["0"]["relative_reduction"], red["9"]["relative_reduction"]
            lines.append(
                f"{ds_label} & {mech_labels[key]} & {v['utility_mean']:.3f}$\\pm${v['utility_std']:.3f} & "
                f"{r0['mean']:+.3f} [{r0['ci_lo']:+.3f}, {r0['ci_hi']:+.3f}] & "
                f"{r9['mean']:+.3f} [{r9['ci_lo']:+.3f}, {r9['ci_hi']:+.3f}] & "
                f"{rrd['mean']:.2f} [{rrd['ci_lo']:.2f}, {rrd['ci_hi']:.2f}] \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("04d_decile_reduction.tex", "\n".join(lines) + "\n")


def gen_04d_inmodel_vs_external_table():
    names = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
    d = json.load(open(ROOT / "reports/figures/phase4d_analysis/inmodel_vs_external.json"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{In-model adversary confidence vs.\ external attacker top-1, plain Mo-PAE, "
        r"deciles 0 and 9. The in-model adversary stays far closer to chance and far flatter "
        r"across deciles than the external attacker on the identical features. Generated from "
        r"\texttt{reports/figures/phase4d\_analysis/inmodel\_vs\_external.json}.}",
        r"\label{tab:inmodel-vs-external}",
        r"\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Dataset & Chance & Decile 0 (ext.\ / in-model) & Decile 9 (ext.\ / in-model) \\",
        r"\midrule",
    ]
    for ds, label in names.items():
        chance = d[ds]["chance_level"]
        d0, d9 = d[ds]["by_decile"]["0"], d[ds]["by_decile"]["9"]
        lines.append(
            f"{label} & {chance:.4f} & {d0['external_top1']:.3f} / {d0['inmodel_confidence']:.4f} & "
            f"{d9['external_top1']:.3f} / {d9['inmodel_confidence']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("04d_inmodel_vs_external.tex", "\n".join(lines) + "\n")


def gen_04e_geolife_10seed_dominance_table():
    d = json.load(open(ROOT / "reports/figures/phase4e_analysis/reference_curve_comparisons.json"))
    tr = d["geolife"]["ten_seed_recheck"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GeoLife, 10 encoder seeds, trained to convergence: paired bootstrap test of "
        r"ed\_soft vs.\ plain Mo-PAE on per-seed utility and per-seed decile-0 risk. "
        r"Dominance requires BOTH a significant utility gain and a significant risk "
        r"reduction; neither $\lambda_4$ value clears this bar.}",
        r"\label{tab:geolife-10seed-dominance}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"$\lambda_4$ & $\Delta$utility (ed\_soft$-$plain) & excl.\ 0? & $\Delta$risk (plain$-$ed\_soft) & excl.\ 0? \\",
        r"\midrule",
    ]
    for l4 in (0.3, 1.0, 3.0):
        v = tr[f"ed_soft_lambda{l4}"]
        ut, rt = v["utility_vs_plain_test"], v["decile0_risk_vs_plain_test"]
        lines.append(
            f"{l4} & {ut['mean_diff']:+.3f} [{ut['ci_lo']:+.3f},{ut['ci_hi']:+.3f}] & "
            f"{'yes' if ut['ci_excludes_zero'] else 'no'} & "
            f"{rt['mean_diff']:+.3f} [{rt['ci_lo']:+.3f},{rt['ci_hi']:+.3f}] & "
            f"{'yes' if rt['ci_excludes_zero'] else 'no'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("04e_geolife_10seed_dominance.tex", "\n".join(lines) + "\n")


def gen_04e_no_privacy_ceiling_table():
    d = json.load(open(ROOT / "reports/figures/phase4e_analysis/reference_curve_comparisons.json"))
    names = {"geolife": "GeoLife", "foursquare": "Foursquare", "gowalla_nyc": "Gowalla-NYC"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{No-privacy ceiling utility, at the original (shorter) training "
        r"budget and once extended to convergence.}",
        r"\label{tab:no-privacy-ceiling}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Dataset & Shorter budget & Converged \\",
        r"\midrule",
    ]
    for ds, label in names.items():
        c20 = d[ds]["no_privacy_ceiling_20ep_stale"]
        c40 = d[ds]["no_privacy_ceiling_40ep"]
        lines.append(
            f"{label} & {c20['utility_mean']:.3f}$\\pm${c20['utility_std']:.3f} & "
            f"{c40['utility_mean']:.3f}$\\pm${c40['utility_std']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("04e_no_privacy_ceiling.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    gen_gowalla_disparate_risk_table()
    gen_main_datasets_disparate_risk_table()
    gen_adaptive_gap_decomposition_table()
    gen_04d_decile_reduction_table()
    gen_04d_inmodel_vs_external_table()
    gen_04e_geolife_10seed_dominance_table()
    gen_04e_no_privacy_ceiling_table()
