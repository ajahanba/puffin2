"""Figure: external-attacker re-identification risk by entropy decile,
no-privacy vs Mo-PAE (lambda3=0.3), four datasets. Reads the per-user CSVs."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "reports/figures/retraining_to_convergence"
DS = [("GeoLife","geolife"),("Foursquare-NYC","foursquare"),("Gowalla (SF Bay)","gowalla_sf_bay"),("YJMob100K","yjmob100k")]
RUNS = [("No privacy","no_privacy","#2a78d6","o"),("Mo-PAE ($\\lambda_3$=0.3)","plain_lp0.3","#eb6834","s")]
plt.rcParams.update({"font.family":"serif","font.size":7,"axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5})
fig, axs = plt.subplots(1,4,figsize=(7.0,1.75),sharex=True)
for ax,(title,ds) in zip(axs,DS):
    for lab,run,c,m in RUNS:
        df = pd.read_csv(f"{R}/{ds}_{run}/per_user_04d.csv"); df = df[df.n_windows>=10]
        g = df.groupby("entropy_decile").post_top1; mu = g.mean(); se = g.std()/np.sqrt(g.size())
        ax.errorbar(mu.index, mu.values, yerr=1.96*se.values, color=c, marker=m, ms=3, lw=1.2, elinewidth=0.6, capsize=0, label=lab)
    ax.set_title(title, fontsize=7.5); ax.set_ylim(0, None); ax.set_xticks([0,3,6,9])
    ax.grid(axis="y", lw=0.3, color="#d9d8d4"); ax.spines[["top","right"]].set_visible(False)
    pass
axs[0].set_ylabel("Re-identification risk")
axs[0].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.supxlabel("Entropy decile (0 = most predictable users)", fontsize=7, y=0.02)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig("paper/figures/decile_risk.pdf"); fig.savefig("paper/figures/decile_risk.png", dpi=200)
