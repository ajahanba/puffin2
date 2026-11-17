"""Print significance (95% CI excludes 0) for PUFFIN rows of the matched table, and the appendix CI table."""
import pandas as pd
c=pd.read_csv("reports/figures/paper/item2_bootstrap_cis_matched_utility.csv")
c["sig"]=(c.ci_lo>0)|(c.ci_hi<0)
N={"geolife":"GeoLife","foursquare":"Foursquare","gowalla_sf_bay":"Gowalla","yjmob100k":"YJMob100K"}
M=[("mean_risk","Mean risk"),("decile0","Most pred."),("decile9","Least pred."),("D_abs_entropy","$D_{\\mathrm{abs}}$ (ent.)"),("D_abs_location_distribution","$D_{\\mathrm{abs}}$ (loc.)")]
rows=[]
for ds in N:
    for cmp,lab in ([(f"ed_soft_entropy_lambda{x}",f"\\puffin{{}} ent., {x}") for x in ["0.3","1.0","3.0"]]+[(f"ed_soft_location_gradmatched_lambda{x}",f"\\puffin{{}} loc., {x}") for x in ["0.3","1.0","3.0"]]+[(f"lipschitz_entropy_gradmatched_lambda{x}",f"Lipschitz ent., {x}") for x in ["0.001","0.003","0.01","0.03","0.1"]]+[(f"lipschitz_location_gradmatched_lambda{x}",f"Lipschitz loc., {x}") for x in ["0.001","0.003","0.01","0.03","0.1"]]+[(f"group_dro_gradmatched_lambda{x}",f"Group DRO, {x}") for x in ["0.1","0.3","1.0","3.0"]]):
        g=c[(c.dataset==ds)&(c.comparison==cmp)].set_index("metric")
        if g.empty: continue
        if g.unresolvable_extrapolation.any(): rows.append(f"{N[ds]} & {lab} & \\multicolumn{{5}}{{c}}{{outside Mo-PAE's utility range}} \\\\"); continue
        cells=[]
        for m,_ in M:
            r=g.loc[m]; cells.append(f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"+("$^*$" if r.sig else ""))
        rows.append(f"{N[ds]} & {lab} & "+" & ".join(cells)+" \\\\")
        print("SIG",ds,cmp,{m:bool(g.loc[m].sig) for m,_ in M})
open("paper/ci_appendix_rows.tex","w").write("\n".join(rows)+"\n")
