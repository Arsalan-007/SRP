"""
Simplified presentation chart for the "Does Attention Earn Its Place?" slide.

Values copied verbatim from v4/02_attention_ablation.ipynb's own run (results table in cell
fa122b52: California Housing, test R^2, mean +/- std over 5 seeds; params and min/run from
the same table). No new numbers -- only a simpler view of four of that notebook's arms.

Run: python3 make_attention_simple.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting import apply_style, role_color, INK, INK_SECONDARY, _strip_axes

apply_style()

ARMS = [
    # label,                                   R2,     std,    color
    ("Average the experts\n(small head, 12k params)", 0.7889, 0.0027, role_color("PlainMLP")),
    ("Average the experts\n(bigger head, 38k params)", 0.8104, 0.0050, role_color("meanpool")),
    ("Bidirectional attention\n(38k params)",      0.8072, 0.0019, role_color("bidirectional")),
    ("Causal attention\n(38k params)",             0.8074, 0.0055, role_color("causal")),
]

fig, ax = plt.subplots(figsize=(11, 5.4))
x = range(len(ARMS))
for i, (lab, m, s, c) in enumerate(ARMS):
    ax.bar(i, m, yerr=s, capsize=5, width=0.62, color=c, edgecolor="white", zorder=3)
    ax.text(i, m + s + 0.0015, f"{m:.4f}", ha="center", fontsize=15, fontweight="bold", color=INK)

# Bracket over the three same-size arms.
y = 0.8215
ax.plot([0.7, 0.7, 3.3, 3.3], [y - 0.0015, y, y, y - 0.0015], color=INK_SECONDARY, lw=1.4)
ax.text(2.0, y + 0.001, "same size — no real difference", ha="center", va="bottom",
        fontsize=13, color=INK_SECONDARY, fontweight="bold")

ax.set_xticks(list(x)); ax.set_xticklabels([a[0] for a in ARMS], fontsize=12)
ax.set_ylabel("Test R²  (higher is better)", fontsize=13)
ax.set_ylim(0.77, 0.828)
ax.set_title("Attention doesn't beat simply averaging the experts at the same size",
             fontweight="bold", fontsize=15)
_strip_axes(ax)
plt.tight_layout()

out_dir = "/home/shayan/Desktop/SRP/figures"
fig.savefig(f"{out_dir}/attention_simple.pdf", dpi=200)
fig.savefig(f"{out_dir}/attention_simple.png", dpi=200)
print("saved to", out_dir)
