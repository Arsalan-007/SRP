"""
plotting.py  —  SRP v4

One shared visual language for every notebook, instead of each of the 14 reinventing its
own matplotlib boilerplate. Surveyed all ~44 plotting cells across notebooks 01-14 before
writing this — the functions below cover the patterns that actually recur (a grouped bar
comparison with error bars appears in nearly every notebook; line sweeps, learning curves,
and the expert-collapse heatmap each appear in several). A few genuinely bespoke, one-off
charts (notebook 02's paired-difference forest plot, its waterfall decomposition, its
attention-weight grid) are NOT forced into this module — three similar lines beat a
premature abstraction used exactly once. Those keep their own code but adopt this module's
PALETTE/typography for visual consistency.

Design source: the `dataviz` skill (validated categorical palette, mark specs, anti-patterns
catalog) — colors below are the skill's reference 8-hue palette
(`references/palette.md`), not picked ad hoc. Direct value labels are used instead of
relying on a legend wherever there are few enough bars for it to stay readable, per the
skill's "selective direct labels" mark spec.

FIXED ROLE PALETTE — the actual point of centralizing this
------------------------------------------------------------
The same color means the same thing in every chart in this repository, always:

    PlainMLP            -> grey     (the floor reference)
    meanpool             -> violet   (the control / simplest option)
    meanpool_wide         -> violet  (same identity, just a capacity-matched variant)
    bidirectional         -> aqua
    causal                -> magenta
    causal_ds              -> red     (darker: "the full recipe")
    XGBoost / GBDT         -> orange   (the target this whole project is chasing)
    V1                      -> blue    (predecessor architecture)
    NCL / meanpool_ncl      -> blue    (Negative Correlation Learning)

Previously every notebook picked its own hex codes per arm (see e.g. notebook 01's inline
PALETTE dict vs notebook 02's inline COLOR dict — same "meanpool" concept, two different
purples). `role_color()` below resolves any of the many arm/model name spellings used
across 14 notebooks (`"meanpool"`, `"WeakEnsemble-MeanPool"`, `"Mean-pool"`, ...) to ONE
consistent color.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

__all__ = ["PALETTE", "INK", "INK_SECONDARY", "GRID", "apply_style", "role_color",
           "plot_bar_comparison", "plot_line_sweep", "plot_learning_curves",
           "plot_expert_collapse_heatmap"]

# ---------------------------------------------------------------------------
# Palette (dataviz skill's validated 8-hue reference set, references/palette.md)
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
GREY = "#9aa0a6"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e3e2dc"

# Canonical role -> color. Every notebook's own arm-name spelling gets mapped onto this
# fixed set by role_color() below, so "meanpool", "Mean-pool", and "WeakEnsemble-MeanPool"
# all draw as the exact same violet everywhere in the repository.
PALETTE = {
    "plainmlp": GREY,
    "v1": BLUE,
    "meanpool": VIOLET,
    "meanpool_wide": VIOLET,
    "bidirectional": AQUA,
    "causal": MAGENTA,
    "causal_ds": RED,
    "meanpool_ncl": BLUE,
    "ncl": BLUE,
    "gate": GREEN,
    "xgboost": ORANGE,
    "gbdt": ORANGE,
    "catboost": ORANGE,
    "none": GREY,
    "linear": AQUA,
    "periodic": VIOLET,
    "plr": VIOLET,
    "piecewise_linear": MAGENTA,
}

_ALIASES = {
    "weakensemble-meanpool": "meanpool", "mean-pool": "meanpool", "mean-pool (wide head)": "meanpool_wide",
    "bidirectional attn": "bidirectional", "causal attn": "causal",
    "causal attn + deep sup.": "causal_ds", "histgradientboosting": "xgboost", "ridge": "plainmlp",
    "xgb": "xgboost",
}


def role_color(name: str, fallback: str = GREY) -> str:
    """Resolve ANY of the many arm/model name spellings used across the 14 notebooks to
    one consistent color. Case/punctuation-insensitive; falls back to grey (not a random
    color) for anything unrecognised, so a new arm name never silently gets a clashing hue."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key in PALETTE:
        return PALETTE[key]
    for role, color in PALETTE.items():
        if role in key:
            return color
    return fallback


def apply_style():
    """Call once per notebook, in place of `sns.set_theme(...)`. No seaborn dependency —
    plain matplotlib rcParams, so every chart looks the same regardless of which notebook
    happens to have seaborn imported."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def _strip_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


# ---------------------------------------------------------------------------
# The workhorse: grouped bar comparison with error bars + direct labels + reference lines.
# Covers: notebook 01 cells 21/23, 02 cell 14, 03 cell 17, 04 cell 10, 06 cell 13, and the
# same pattern reimplemented ad hoc in 08-14.
# ---------------------------------------------------------------------------
def plot_bar_comparison(labels, means, stds=None, colors=None, seed_values=None,
                        ref_lines=None, horizontal=False, ylabel=None, title=None,
                        value_fmt="{:.4f}", ax=None, figsize=(10, 4.8)):
    """One bar per label, optional error bars, optional raw per-seed dots overlaid, optional
    dashed reference lines (e.g. "our XGBoost", "published FT-Transformer") with inline text
    labels instead of a legend entry each. Direct value labels above/beside every bar.

    labels        : list[str] — x tick labels (or y labels if horizontal=True).
    means         : list[float] — bar heights.
    stds          : list[float] | None — error bar half-widths.
    colors        : list[str] | None — defaults to role_color(label) for each label.
    seed_values   : list[list[float]] | None — raw per-seed values, plotted as jittered dots.
    ref_lines     : list[(value, label, color)] | None — dashed horizontal/vertical lines.
    horizontal    : bool — horizontal bars (good for >6 labels or long label text).
    """
    means = np.asarray(means, dtype=float)
    stds = np.zeros_like(means) if stds is None else np.asarray(
        [0.0 if s is None else s for s in stds], dtype=float)
    colors = colors or [role_color(l) for l in labels]
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    x = np.arange(len(labels))

    if horizontal:
        ax.barh(x, means, xerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_yticks(x); ax.set_yticklabels(labels); ax.invert_yaxis()
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + (means.max() * 0.012), i, value_fmt.format(m),
                    va="center", fontsize=10, fontweight="bold", color=INK)
        if ylabel:
            ax.set_xlabel(ylabel)
    else:
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
        if seed_values is not None:
            for i, ys in enumerate(seed_values):
                ys = np.asarray(ys, dtype=float)
                jitter = np.linspace(-0.12, 0.12, len(ys)) if len(ys) > 1 else [0.0]
                ax.scatter(np.full(len(ys), i) + jitter, ys, s=16, color=INK, alpha=0.55, zorder=4)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + (means.max() - means.min() + 1e-9) * 0.02, value_fmt.format(m),
                    ha="center", fontsize=10, fontweight="bold", color=INK)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel)

    for val, label, color in (ref_lines or []):
        line_fn = ax.axvline if horizontal else ax.axhline
        line_fn(val, color=color, linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
        if horizontal:
            # Blended transform: x in DATA coords (the line's value), y anchored near the
            # axes' top edge in AXES-FRACTION coords — stays inside the figure regardless
            # of how many bars there are (a hardcoded offset would not).
            ax.text(val, 0.98, label, transform=ax.get_xaxis_transform(), color=color,
                   fontsize=9.5, fontweight="bold", ha="center", va="top")
        else:
            # Same idea, axes other way round: x anchored near the LEFT edge of the axes
            # (axes-fraction), y at the line's actual DATA value. Left, not right, because
            # these charts are conventionally ordered worst-to-best left-to-right, so a
            # "best baseline" reference line sits closest to the rightmost bars' own value
            # labels — anchoring right caused real overlap (e.g. notebook 01's headline
            # chart, "Best (GBDT)" colliding with the XGBoost bar's "0.8471" label).
            ax.text(0.015, val, label, transform=ax.get_yaxis_transform(), color=color,
                   fontsize=9.5, fontweight="bold", va="bottom", ha="left")

    if title:
        ax.set_title(title)
    _strip_axes(ax)
    if fig is not None:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Line sweep: value vs. a swept quantity (lambda, K, epoch position, Optuna trial, ...).
# Covers: notebook 03 cells 12/19, notebook 11's lambda sweep, notebook 02 cell 26.
# ---------------------------------------------------------------------------
def plot_line_sweep(x, series, ref_lines=None, band=True, xlabel=None, ylabel=None,
                    title=None, ax=None, figsize=(9.5, 5.2), annotate_last=None):
    """series: dict[label -> (means, stds_or_None)] OR dict[label -> means] (no band).
    One line per series, colored via role_color(label), optional shaded +/- std band,
    optional dashed reference lines, optional direct annotation on the last point of one
    named series (annotate_last: a label present in `series`)."""
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    x = np.asarray(x, dtype=float)

    for label, vals in series.items():
        means, stds = (vals if isinstance(vals, tuple) else (vals, None))
        means = np.asarray(means, dtype=float)
        color = role_color(label)
        if band and stds is not None:
            stds = np.asarray([0.0 if s is None else s for s in stds], dtype=float)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(x, means, color=color, linewidth=2.2, marker="o", markersize=6,
               markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)

    for val, label, color in (ref_lines or []):
        ax.axhline(val, color=color, linestyle=(0, (5, 3)), linewidth=1.6, zorder=1)
        # Inline text, same as plot_bar_comparison's ref lines -- otherwise a reader has no
        # way to tell two dashed lines apart except by color. Left edge, axes-fraction x /
        # data-value y, so it stays in bounds regardless of the x-axis range.
        ax.text(0.015, val, label, transform=ax.get_yaxis_transform(), color=color,
               fontsize=9.5, fontweight="bold", va="bottom", ha="left")

    if annotate_last and annotate_last in series:
        means, _ = (series[annotate_last] if isinstance(series[annotate_last], tuple)
                   else (series[annotate_last], None))
        ax.annotate(f"{annotate_last}: {means[-1]:.4f}", xy=(x[-1], means[-1]),
                   xytext=(8, 0), textcoords="offset points", fontsize=10,
                   fontweight="bold", color=role_color(annotate_last), va="center")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1:
        ax.legend(fontsize=9.5, loc="best")
    _strip_axes(ax)
    if fig is not None:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Learning curves: train/val loss vs epoch, one or more arms overlaid.
# Covers: notebook 01 cell 25.
# ---------------------------------------------------------------------------
def plot_learning_curves(histories, xlabel="epoch", ylabel="loss", title=None,
                         ax=None, figsize=(9.5, 5.0)):
    """histories: dict[label -> history] where history is EITHER a single
    {"train": [...], "val": [...]} dict, OR a list of such dicts (e.g. one per seed) — all
    drawn in that label's color, only the first contributing a legend entry. Faint = train,
    solid = val, a dotted vertical line marks each curve's best (lowest-val) epoch — the
    point early stopping actually restores, which is the thing worth seeing here."""
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    for label, h in histories.items():
        color = role_color(label)
        runs = h if isinstance(h, list) else [h]
        for i, run in enumerate(runs):
            train, val = run.get("train"), run.get("val")
            if train is not None:
                ax.plot(train, color=color, alpha=0.35, linewidth=1.2)
            if val is not None:
                ax.plot(val, color=color, alpha=0.95, linewidth=2.0 if len(runs) == 1 else 1.4,
                       label=label if i == 0 else None)
                best_epoch = int(np.argmin(val))
                ax.axvline(best_epoch, color=color, linestyle=(0, (1, 2)), linewidth=1.2, alpha=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9.5, loc="best")
    _strip_axes(ax)
    if fig is not None:
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Expert collapse heatmap: pairwise cosine similarity, before vs after training, + a
# summary bar. Covers: notebook 01 cell 29 (named function, as requested).
# ---------------------------------------------------------------------------
def plot_expert_collapse_heatmap(sim_before, sim_after, off_before=None, off_after=None,
                                 figsize=(13, 4.2)):
    """sim_before/sim_after: (K, K) pairwise cosine similarity matrices. off_before/
    off_after: mean off-diagonal values (arrays over seeds, or single floats) — if not
    given, computed directly from the two matrices. Three panels: before heatmap, after
    heatmap (shared color scale), and a before/after summary bar with a "total collapse"
    reference line at 1.0."""
    sim_before, sim_after = np.asarray(sim_before), np.asarray(sim_after)
    if off_before is None:
        K = sim_before.shape[0]
        off_before = sim_before[~np.eye(K, dtype=bool)].mean()
    if off_after is None:
        K = sim_after.shape[0]
        off_after = sim_after[~np.eye(K, dtype=bool)].mean()
    off_before_arr = np.atleast_1d(off_before)
    off_after_arr = np.atleast_1d(off_after)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    vmax = max(np.abs(sim_before).max(), np.abs(sim_after).max())
    for ax, mat, title, off in [(axes[0], sim_before, "BEFORE training (random init)", off_before_arr.mean()),
                                (axes[1], sim_after, "AFTER training", off_after_arr.mean())]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"{title}\noff-diag mean = {off:+.3f}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("expert j"); ax.set_ylabel("expert i")
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    means = [off_before_arr.mean(), off_after_arr.mean()]
    stds = [off_before_arr.std(ddof=1) if len(off_before_arr) > 1 else 0.0,
           off_after_arr.std(ddof=1) if len(off_after_arr) > 1 else 0.0]
    ax.bar(["before\ntraining", "after\ntraining"], means, yerr=stds, capsize=6,
          color=[GREY, VIOLET], edgecolor="white", linewidth=1.5)
    ax.axhline(1.0, color=RED, linestyle=(0, (5, 3)), linewidth=1.4)
    ax.text(1.4, 0.95, "total collapse", color=RED, ha="right", fontsize=9.5)
    ax.set_ylim(-0.2, 1.1)
    ax.set_ylabel("mean off-diagonal cosine similarity")
    ax.set_title("Expert collapse check\n(mean ± std over seeds)", fontsize=10.5, fontweight="bold")
    for i, v in enumerate(means):
        ax.text(i, v + 0.05, f"{v:+.3f}", ha="center", fontweight="bold", color=INK)
    _strip_axes(ax)
    plt.tight_layout()
    return axes


# ---------------------------------------------------------------------------
# Smoke test:  python plotting.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    apply_style()

    print("CHECKS")

    # 1. role_color resolves every alias spelling seen across the 14 notebooks to ONE color.
    spellings = ["meanpool", "Mean-pool", "WeakEnsemble-MeanPool", "mean-pool (wide head)"]
    colors = {role_color(s) for s in spellings}
    print(f"  [{'PASS' if len(colors) == 1 else 'FAIL'}] every 'meanpool' spelling "
          f"{spellings} resolves to the same color ({colors})")

    ok = role_color("XGBoost") == role_color("xgboost") == role_color("HistGradientBoosting") == PALETTE["xgboost"]
    print(f"  [{'PASS' if ok else 'FAIL'}] XGBoost-family names all resolve to the palette's orange")

    ok = role_color("some_never_seen_arm_xyz") == GREY
    print(f"  [{'PASS' if ok else 'FAIL'}] an unrecognised name falls back to grey, not a random hue")

    # 2. Each plotting function runs end-to-end on realistic shapes and produces a real figure.
    np.random.seed(0)
    ax = plot_bar_comparison(
        ["PlainMLP", "meanpool", "causal", "XGBoost"],
        means=[0.51, 0.54, 0.52, 0.43], stds=[0.01, 0.008, 0.012, 0.002],
        seed_values=[[0.50, 0.52, 0.51], [0.53, 0.55, 0.54], [0.51, 0.53, 0.52], [0.428, 0.431, 0.430]],
        ref_lines=[(0.459, "published FT-Transformer", ORANGE)],
        ylabel="RMSE", title="smoke test: bar comparison")
    ok = ax is not None and len(ax.patches) == 4
    print(f"  [{'PASS' if ok else 'FAIL'}] plot_bar_comparison renders 4 bars + seed dots + a ref line")
    plt.close("all")

    ax = plot_line_sweep(
        x=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        series={"meanpool_ncl": ([0.45, 0.44, 0.43, 0.42, 0.41, 0.40], [0.01] * 6)},
        ref_lines=[(0.4328, "our XGBoost", ORANGE)],
        xlabel="lambda", ylabel="RMSE", title="smoke test: line sweep", annotate_last="meanpool_ncl")
    ok = ax is not None and len(ax.lines) >= 1
    print(f"  [{'PASS' if ok else 'FAIL'}] plot_line_sweep renders a sweep curve + annotation + ref line")
    plt.close("all")

    hist = {"meanpool": {"train": list(np.linspace(1.0, 0.3, 40) + np.random.randn(40) * 0.02),
                        "val": list(np.linspace(1.1, 0.35, 40) + np.random.randn(40) * 0.03)},
           "causal": {"train": list(np.linspace(1.0, 0.28, 40) + np.random.randn(40) * 0.02),
                      "val": list(np.linspace(1.1, 0.38, 40) + np.random.randn(40) * 0.03)}}
    ax = plot_learning_curves(hist, title="smoke test: learning curves")
    ok = ax is not None and len(ax.lines) == 6   # 2 arms x (train + val + best-epoch marker)
    print(f"  [{'PASS' if ok else 'FAIL'}] plot_learning_curves renders train+val+best-epoch-marker "
          f"for 2 arms ({len(ax.lines)} lines)")
    plt.close("all")

    # Multi-seed variant (notebook 01's actual use case: several seeds per arm, one plot).
    multi_hist = {"meanpool": [{"train": list(np.linspace(1.0, 0.3, 20)),
                               "val": list(np.linspace(1.1, 0.35, 20))} for _ in range(3)]}
    ax = plot_learning_curves(multi_hist, title="smoke test: multi-seed learning curves")
    ok = ax is not None and len(ax.lines) == 9   # 3 seeds x (train + val + best-epoch marker)
    print(f"  [{'PASS' if ok else 'FAIL'}] plot_learning_curves (multi-seed list form) renders "
          f"3 seeds for 1 arm ({len(ax.lines)} lines), single legend entry "
          f"({'PASS' if len(ax.get_legend().get_texts()) == 1 else 'FAIL'})")
    plt.close("all")

    K = 8
    sim_before = np.eye(K) + np.random.randn(K, K) * 0.02
    sim_after = np.eye(K) * 0.9 + np.random.rand(K, K) * 0.15
    axes = plot_expert_collapse_heatmap(sim_before, sim_after)
    ok = len(axes) == 3
    print(f"  [{'PASS' if ok else 'FAIL'}] plot_expert_collapse_heatmap renders 3 panels "
          f"(before heatmap, after heatmap, summary bar)")
    plt.close("all")

    print("\n(all smoke-test figures closed; run interactively / in a notebook to actually view them)")
