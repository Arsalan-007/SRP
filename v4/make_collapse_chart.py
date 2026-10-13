"""
Slide-sized version of notebook 16's expert-collapse chart (larger fonts, shorter height).

Per-seed values copied verbatim from v4/16_feature_bagging_collapse.ipynb's own run
(500-epoch cap, every run converged): mean pairwise cosine similarity between the 16 experts'
representations on the validation set, after training. 1.0 = all experts identical.

Run: python3 make_collapse_chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import apply_style, role_color, INK, _strip_axes

apply_style()

SIM = {  # dataset -> (with bagging, without bagging), one value per seed 0..4
    "California Housing": ([0.069, 0.071, 0.069, 0.094, 0.058], [0.103, 0.079, 0.088, 0.093, 0.093]),
    "Higgs Small":        ([0.038, 0.031, 0.030, 0.044, 0.039], [0.045, 0.055, 0.046, 0.049, 0.062]),
}
colors = [role_color("meanpool"), role_color("PlainMLP")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, (name, arms) in zip(axes, SIM.items()):
    for i, vals in enumerate(arms):
        v = np.asarray(vals)
        ax.bar(i, v.mean(), width=0.55, color=colors[i], zorder=3)
        ax.text(i, v.mean() + 0.04, f"{v.mean():.2f}", ha="center", fontsize=17,
               fontweight="bold", color=INK)
    ax.axhline(1.0, color="#d33", linestyle=(0, (5, 3)), linewidth=2)
    ax.text(0.5, 0.95, "total collapse = 1.0 (all experts identical)", transform=ax.get_yaxis_transform(),
           color="#d33", fontsize=13, fontweight="bold", ha="center", va="top")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["with feature\nbagging", "without feature\nbagging"], fontsize=14)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(name, fontweight="bold", fontsize=16)
    _strip_axes(ax)
axes[0].set_ylabel("how alike the experts are", fontsize=13)
plt.tight_layout()

out_dir = "/home/shayan/Desktop/SRP/figures"
fig.savefig(f"{out_dir}/feature_bagging_collapse_slide.pdf", dpi=200)
fig.savefig(f"{out_dir}/feature_bagging_collapse_slide.png", dpi=200)
print("saved to", out_dir)
