"""
Standalone presentation-chart generator for the "Does Attention Earn Its Place?" slide.

Values are copied verbatim from v4/02_attention_ablation.ipynb's own actual run (Test R2
means from cell fa122b52's results table, and the paired-difference deltas from cell
0cf8cded's decomposition table -- both re-executed for real in this session, not estimated).
This script only re-renders those already-real numbers at presentation quality/size.

Run: python3 make_attention_waterfall.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import apply_style, role_color, INK, _strip_axes

apply_style()

# Verbatim from notebook 02's own results (California Housing, test R^2, 5 seeds).
BASE = 0.7889                 # Mean-pool
STEPS = [
    ("+ capacity",              +0.0215),   # wide head, same params as attention arms
    ("+ attention\n(no order)", -0.0032),   # bidirectional
    ("+ causal\nordering",      +0.0002),   # causal - bidirectional
    ("+ deep\nsupervision",     +0.0023),   # causal_ds - causal
]
TOTAL = 0.8096                # Causal attn + deep sup. (= BASE + sum(deltas), reported directly)
PLAINMLP = 0.8030
XGBOOST = 0.8471

labels = ["Mean-pool"] + [s[0] for s in STEPS] + ["= Causal attn\n+ deep sup."]
fig, ax = plt.subplots(figsize=(11.5, 5.6))

level = BASE
for i, (label, delta) in enumerate(STEPS, start=1):
    color = "#1a9850" if delta >= 0 else "#d73027"
    bottom = level if delta >= 0 else level + delta
    ax.bar(i, abs(delta), bottom=bottom, width=0.6, color=color, zorder=3)
    top = level + max(delta, 0)
    ax.text(i, top + 0.0018, f"{delta:+.4f}", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.plot([i - 0.3, i + 0.3], [level + delta, level + delta], color="#999", lw=0.9, ls=":", zorder=1)
    level += delta

# Base and total bars.
ax.bar(0, BASE, width=0.6, color=role_color("meanpool"), zorder=3)
ax.text(0, BASE + 0.0018, f"{BASE:.4f}", ha="center", fontsize=11.5, fontweight="bold", color=INK)
ax.bar(len(STEPS) + 1, TOTAL, width=0.6, color=role_color("causal_ds"), zorder=3)
ax.text(len(STEPS) + 1, TOTAL + 0.0018, f"{TOTAL:.4f}", ha="center", fontsize=11.5, fontweight="bold", color=INK)

ax.axhline(PLAINMLP, color=role_color("PlainMLP"), linestyle="--", linewidth=1.4, zorder=2)
ax.text(0.015, PLAINMLP, f"PlainMLP {PLAINMLP:.4f}", transform=ax.get_yaxis_transform(),
       color=role_color("PlainMLP"), fontsize=10.5, fontweight="bold", va="bottom", ha="left")
ax.axhline(XGBOOST, color=role_color("XGBoost"), linestyle="--", linewidth=1.4, zorder=2)
ax.text(0.015, XGBOOST, f"XGBoost {XGBOOST:.4f}", transform=ax.get_yaxis_transform(),
       color=role_color("XGBoost"), fontsize=10.5, fontweight="bold", va="bottom", ha="left")

ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylabel("Test R²", fontsize=12.5)
ax.set_ylim(min(BASE, level) - 0.015, XGBOOST + 0.012)
ax.set_title("Where the accuracy comes from: capacity helps, attention doesn't add more",
             fontweight="bold", fontsize=14)
_strip_axes(ax)
plt.tight_layout()

out_dir = "/home/shayan/Desktop/SRP/figures"
fig.savefig(f"{out_dir}/attention_waterfall.pdf", dpi=200)
fig.savefig(f"{out_dir}/attention_waterfall.png", dpi=200)
print("saved to", out_dir)
