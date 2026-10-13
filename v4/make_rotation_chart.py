"""
Standalone presentation-chart generator for the "Sub-Problem 2: Rotation Invariance" slide.

Does NOT re-run notebook 04 (216 training runs, ~30 min) -- the values below are copied
verbatim from that notebook's own actual printed findings output (cell id 2daaff88, section
"2. ROTATION (USES FEATURE AXES)"), which is itself computed from the real per-seed runs in
cell a93b0627, never hand-typed at the time. This script only re-renders those already-real
numbers with clean styling; it introduces no new claims.

Run: python3 make_rotation_chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import apply_style, role_color, INK, _strip_axes

apply_style()

# Verbatim from v4/04_grinsztajn_diagnostics.ipynb, cell 2daaff88's printed output.
# Delta = accuracy under all-features-rotated minus accuracy at baseline, same model/seed,
# mean over 3 seeds (see notebook 04 section 5 "Dose-response" / cell a93b0627).
DATASETS = ["Covertype", "Poker Hand", "HIGGS"]
XGB_DELTA = [-0.0225, -0.1763, -0.0926]
OURS_DELTA = [+0.0021, -0.0030, -0.0026]     # v4 mean-pool (wide head), "un-embedded" -- the slide's "our neural model"

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(DATASETS))
w = 0.32

b1 = ax.bar(x - w / 2, XGB_DELTA, w, color=role_color("XGBoost"), edgecolor="white", linewidth=1.2,
           label="XGBoost (tree-based)", zorder=3)
b2 = ax.bar(x + w / 2, OURS_DELTA, w, color=role_color("meanpool"), edgecolor="white", linewidth=1.2,
           label="our model, no embedding", zorder=3)

for bars in (b1, b2):
    for rect in bars:
        v = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, v + (0.004 if v >= 0 else -0.004),
               f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
               fontsize=10.5, fontweight="bold", color=INK)

ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=12)
ax.set_ylabel("Δ test accuracy when features are rotated\n(0 = no effect, negative = hurt)")
ax.set_title("Trees rely on the feature axes -- our model without embeddings doesn't",
             fontweight="bold", fontsize=13.5)
ax.set_ylim(min(XGB_DELTA) - 0.02, 0.03)
ax.legend(fontsize=10.5, loc="upper right")
_strip_axes(ax)
plt.tight_layout()

out_dir = "/home/shayan/Desktop/SRP/figures"
fig.savefig(f"{out_dir}/rotation_invariance.pdf", dpi=200)
fig.savefig(f"{out_dir}/rotation_invariance.png", dpi=200)
print("saved to", out_dir)
