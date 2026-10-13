"""
Presentation chart for "The Strongest Result: NCL on California Housing".

Values copied verbatim from v4/11_ncl_meanpool.ipynb's own run (test RMSE, mean +/- std over
3 seeds, published California Housing split). No new numbers.

Run: python3 make_ncl_chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import apply_style, INK, INK_SECONDARY, BLUE, ORANGE, AQUA, _strip_axes

apply_style()

LAMBDA = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
RMSE = np.array([0.4501, 0.4465, 0.4435, 0.4358, 0.4273, 0.4189, 0.4039])
SD = np.array([0.0081, 0.0076, 0.0079, 0.0077, 0.0022, 0.0026, 0.0039])
DIVERGED = 1.0335          # lambda = 1.5, off-chart
BASELINE = 0.4202          # same model without NCL (embedding-level pooling)
XGB = 0.4328               # our tuned XGBoost

fig, ax = plt.subplots(figsize=(9.4, 5.4))
ax.fill_between(LAMBDA, RMSE - SD, RMSE + SD, color=BLUE, alpha=0.12, linewidth=0)
ax.plot(LAMBDA, RMSE, color=BLUE, linewidth=2.4, marker="o", markersize=7,
        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
ax.scatter([1.0], [RMSE[-1]], s=170, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
ax.text(0.03, 0.4585, "our model + NCL", color=BLUE, fontsize=14, fontweight="bold")

ax.axhline(XGB, color=AQUA, linestyle=(0, (5, 3)), linewidth=1.6)
ax.text(1.08, XGB + 0.002, f"our XGBoost (tuned): {XGB:.4f}", color=AQUA, fontsize=12,
        fontweight="bold", va="bottom")
ax.axhline(BASELINE, color=ORANGE, linestyle=(0, (5, 3)), linewidth=1.6)
ax.text(1.08, BASELINE - 0.004, f"our model without NCL: {BASELINE:.4f}", color=ORANGE,
        fontsize=12, fontweight="bold", va="top")

ax.annotate("λ=1.0: RMSE 0.4039\n(best result in the project)", xy=(1.0, RMSE[-1]),
            xytext=(0.62, 0.357), fontsize=12.5, fontweight="bold", color=INK,
            arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=1.0))
ax.annotate(f"λ=1.5 diverges to\nRMSE {DIVERGED:.2f} (off-chart)", xy=(1.55, 0.472),
            xytext=(1.2, 0.466), fontsize=11, style="italic", color=INK_SECONDARY, ha="center",
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1.0,
                            connectionstyle="arc3,rad=-0.3"))

ax.set_xlim(-0.05, 1.55)
ax.set_ylim(0.335, 0.475)
ax.set_xlabel("NCL penalty strength λ", fontsize=13)
ax.set_ylabel("Test RMSE (lower is better)", fontsize=13)
_strip_axes(ax)
plt.tight_layout()

out_dir = "/home/shayan/Desktop/SRP/figures"
fig.savefig(f"{out_dir}/ncl_california_housing.pdf", dpi=200)
fig.savefig(f"{out_dir}/ncl_california_housing.png", dpi=200)
print("saved to", out_dir)
