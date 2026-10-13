"""
Teaching figure for "what does rotating the features actually mean?".

Pure illustration on a synthetic 2-feature loan dataset -- no project results here.
The left panel is the rule a tree can cut with one split; the right panel is the SAME
data and the SAME rule after a 45-degree rotation, where the tree needs a staircase.

Run: python3 make_rotation_explainer.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import apply_style, INK, INK_SECONDARY, BLUE, ORANGE, _strip_axes

apply_style()
rng = np.random.default_rng(0)

N = 260
age = rng.uniform(20, 70, N)
income = rng.uniform(10, 95, N)
approved = income > 50                      # the true rule: Age is irrelevant

theta = np.deg2rad(45)
cx, cy = age.mean(), income.mean()
a, i = age - cx, income - cy
u = a * np.cos(theta) - i * np.sin(theta)   # rotated coordinates
v = a * np.sin(theta) + i * np.cos(theta)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
sc = dict(s=26, alpha=0.85, linewidth=0.5, edgecolor="white", zorder=3)

ax = axes[0]
ax.scatter(age[approved], income[approved], color=BLUE, label="approved", **sc)
ax.scatter(age[~approved], income[~approved], color=ORANGE, label="declined", **sc)
ax.axhline(50, color=INK, linewidth=2.4, zorder=4)
ax.text(21, 52.5, "ONE split:  Income > 50k", fontsize=13, fontweight="bold", color=INK)
ax.set_xlabel("Age", fontsize=13)
ax.set_ylabel("Income (k)", fontsize=13)
ax.set_title("Original features\nA tree cuts this perfectly with one horizontal line",
             fontweight="bold", fontsize=13.5)
ax.legend(fontsize=11, loc="lower right")
_strip_axes(ax)

ax = axes[1]
ax.scatter(u[approved], v[approved], color=BLUE, **sc)
ax.scatter(u[~approved], v[~approved], color=ORANGE, **sc)

# The true boundary, now diagonal in the rotated space.
lim = np.array([min(u.min(), v.min()) - 4, max(u.max(), v.max()) + 4])
b = (50 - cy)
ax.plot(lim, (b - lim * np.sin(theta)) / np.cos(theta), color=INK, linewidth=2.4,
        zorder=4, label="true boundary (diagonal)")

# A tree's staircase approximation of that diagonal.
steps = np.linspace(lim[0], lim[1], 11)
sx, sy = [], []
for k in range(len(steps) - 1):
    y0 = (b - steps[k] * np.sin(theta)) / np.cos(theta)
    y1 = (b - steps[k + 1] * np.sin(theta)) / np.cos(theta)
    sx += [steps[k], steps[k + 1], steps[k + 1]]
    sy += [y0, y0, y1]
ax.plot(sx, sy, color="#d33", linewidth=2.0, linestyle=(0, (4, 2)), zorder=5,
        label="what a tree can draw")

ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("u = blend of Age and Income", fontsize=13)
ax.set_ylabel("v = another blend", fontsize=13)
ax.set_title("Same data, rotated 45°\nThe tree now needs a staircase of many splits",
             fontweight="bold", fontsize=13.5)
ax.legend(fontsize=11, loc="lower left")
_strip_axes(ax)

fig.suptitle("Rotating the features does not change the data — only the coordinate system",
             fontsize=15, fontweight="bold", color=INK_SECONDARY, y=1.01)
plt.tight_layout()

out_dir = "/home/shayan/Desktop/SRP/figures"
fig.savefig(f"{out_dir}/rotation_explained.pdf", dpi=200, bbox_inches="tight")
fig.savefig(f"{out_dir}/rotation_explained.png", dpi=200, bbox_inches="tight")
print("saved to", out_dir)
