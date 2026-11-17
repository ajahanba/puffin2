"""Trade-off figure: utility (Acc@1) vs risk of the most predictable decile.
Inputs: paper/data/results_main.csv, paper/data/results_yjmob.csv (computed from per-user CSVs)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
d = pd.read_csv("paper/data/results_main.csv")
y = pd.read_csv("paper/data/results_yjmob.csv")
y["ds"]="yjmob100k"; y["run"]=y.run.str.replace("plain_lp","Mo-PAE lp").str.replace("ed_soft_lambda","PUFFIN ").str.replace("no_privacy","no-privacy")
d = pd.concat([d,y],ignore_index=True)
SER = [  # label, prefix, color, marker
 ("PUFFIN (entropy)","PUFFIN ","#eb6834","s"),
 ("PUFFIN (location, scaled)","LocGM ","#e87ba4","D"),
 ("Lipschitz (entropy)","LipGM-ent ","#2a78d6","^"),
 ("Lipschitz (location)","LipGM-loc ","#1baf7a","v"),
 ("Group DRO","DROGM ","#6250d6","P"),
 ("LSTM-TrajGAN","TrajGAN","#5a5955","X"),
]
plt.rcParams.update({"font.family":"serif","font.size":7,"axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5})
DS=[("GeoLife","geolife"),("Foursquare-NYC","foursquare"),("Gowalla (SF Bay)","gowalla_sf_bay"),("YJMob100K","yjmob100k")]
fig,axs=plt.subplots(1,4,figsize=(7.0,2.0))
def draw(ax,g,small=False):
    m=g[g.run.str.startswith("Mo-PAE")].copy(); m["lp"]=m.run.str.extract(r"lp([\d.]+)")[0].astype(float); m=m.sort_values("lp")
    ax.plot(m.util,m.dec0,"-o",color="#0b0b0b",ms=2.5,lw=1,label="Mo-PAE ($\\lambda_3$ curve)",zorder=3)
    gi=g[g.run.str.startswith("GI-DP")]
    if len(gi): gi=gi.assign(e=gi.run.str.split().str[1].astype(float)).sort_values("e"); ax.plot(gi.util,gi.dec0,"--o",color="#eda100",ms=2.5,lw=1,label="Geo-indistinguishability",zorder=2)
    n=g[g.run=="no-privacy"]; ax.plot(n.util,n.dec0,"*",color="#8a8984",ms=7,label="No privacy",zorder=4)
    for lab,p,c,mk in SER:
        s=g[g.run.str.startswith(p)]
        if len(s): ax.plot(s.util,s.dec0,mk,color=c,ms=4,mec="white",mew=0.4,ls="none",label=lab,zorder=5)
    ax.grid(lw=0.3,color="#d9d8d4"); ax.spines[["top","right"]].set_visible(False)
for ax,(t,ds) in zip(axs,DS):
    g=d[d.ds==ds]; draw(ax,g); ax.set_title(t,fontsize=7.5)
    if ds=="geolife":
        ia=ax.inset_axes([0.36,0.06,0.40,0.42])
        draw(ia,g); ia.set_xlim(0.68,0.765); ia.set_ylim(0.18,0.36); ia.tick_params(labelsize=5,length=1.5)
        mark_inset(ax,ia,loc1=2,loc2=4,fc="none",ec="#8a8984",lw=0.4)
axs[0].set_ylabel("Risk, most predictable decile")
fig.supxlabel("Utility (Acc@1, next-location prediction)",fontsize=7,y=0.0)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,frameon=False,fontsize=6.3,bbox_to_anchor=(0.5,1.07))
fig.tight_layout(pad=0.3,w_pad=0.8,rect=(0,0.03,1,0.86))
fig.savefig("paper/figures/tradeoff.pdf",bbox_inches="tight"); fig.savefig("paper/figures/tradeoff.png",dpi=200,bbox_inches="tight")
