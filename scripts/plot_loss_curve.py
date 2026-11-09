import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("reports/figures/paper/trajgan_loss_curve_geolife.json"))
dl, gl = d["d_loss_history"], d["g_loss_history"]
epochs = list(range(1, len(dl) + 1))

EPOCH_BUDGET = 80

fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
axes[0].plot(epochs, dl, color="tab:blue", linewidth=1)
axes[0].axvline(EPOCH_BUDGET, color="black", linestyle="--", linewidth=1, label=f"chosen budget (epoch {EPOCH_BUDGET})")
axes[0].axvspan(99, len(dl), color="red", alpha=0.08, label="discriminator-dominance divergence (epoch 99+)")
axes[0].set_ylabel("discriminator loss")
axes[0].legend(fontsize=7, loc="upper right")
axes[0].set_title("GeoLife LSTM-TrajGAN loss curve (seed 0)")

axes[1].plot(epochs, gl, color="tab:orange", linewidth=1)
axes[1].axvline(EPOCH_BUDGET, color="black", linestyle="--", linewidth=1)
axes[1].axvspan(99, len(dl), color="red", alpha=0.08)
axes[1].set_ylabel("generator loss")
axes[1].set_xlabel("epoch")

plt.tight_layout()
out_path = "reports/figures/paper/trajgan_loss_curve_geolife.pdf"
plt.savefig(out_path)
print(f"wrote {out_path}")

# Also save a PNG for quick viewing
plt.savefig("reports/figures/paper/trajgan_loss_curve_geolife.png", dpi=150)
print("wrote reports/figures/paper/trajgan_loss_curve_geolife.png")
