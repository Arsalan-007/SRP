# ...existing code...
import json
import os
import argparse
from typing import List, Dict, Tuple
import math
import statistics

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_FILES = {
    "wine": "/media/homes/hamza/results/n_mlps_averaged_exp/tabjoint_search_results_wine.json",
    "adult": "/media/homes/hamza/results/n_mlps_averaged_exp/tabjoint_search_results_adult.json",
    "churn_modelling": "/media/homes/hamza/results/n_mlps_averaged_exp/tabjoint_search_results_Churn_Modeling.json",
    "california_housing": "/media/homes/hamza/results/n_mlps_averaged_exp/tabjoint_search_results_califonria_housing.json",
}

OUT_HTML = "/media/homes/hamza/results/n_mlps_vs_accuracy_multi.html"
OUT_PNG = "/media/homes/hamza/results/n_mlps_vs_accuracy_multi.png"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def load_results_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    return [data]

def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(0,0,0,{alpha})"

def extract_n_mlps_and_accuracy_with_std(results: List[dict]) -> Tuple[List[int], List[float], List[float], List[int]]:
    """
    Returns sorted lists: n_mlps, best_val_mean, best_val_std, num_runs
    missing std -> 0.0, missing num_runs -> 1
    """
    records = []
    for r in results:
        n = None
        if "n_mlps" in r:
            n = r["n_mlps"]
        else:
            hp = r.get("hparams")
            if isinstance(hp, dict) and "n_mlps" in hp:
                n = hp["n_mlps"]
        if n is None:
            continue

        scores = r.get("scores", {})
        if isinstance(scores, dict):
            mean = scores.get("best_val_mean", scores.get("best_val"))
            std = scores.get("best_val_std", 0.0)
            runs = scores.get("num_runs", 1)
        else:
            mean = r.get("best_val_mean", r.get("best_val"))
            std = r.get("best_val_std", 0.0)
            runs = r.get("num_runs", 1)

        if mean is None:
            continue
        try:
            records.append((int(n), float(mean), float(std or 0.0), int(runs or 1)))
        except Exception:
            continue

    if not records:
        return [], [], [], []
    records.sort(key=lambda x: x[0])
    xs = [t[0] for t in records]
    ys = [t[1] for t in records]
    stds = [t[2] for t in records]
    runs = [t[3] for t in records]
    return xs, ys, stds, runs

def compute_plot_std(std: float, num_runs: int) -> float:
    """
    Show reduced uncertainty for plotting:
    - if num_runs>1 use standard error = std / sqrt(n)
    - else fallback to shrink factor 0.5
    """
    try:
        if num_runs and num_runs > 1:
            return std / math.sqrt(num_runs)
    except Exception:
        pass
    return std * 0.5

def build_figure(datasets: Dict[str, str], colors: List[str]) -> go.Figure:
    names = list(datasets.keys())
    fig = make_subplots(rows=2, cols=2, subplot_titles=names)
    positions = [(1,1), (1,2), (2,1), (2,2)]
    tick_vals = list(range(2, 17, 2))  # 2,4,6,...,16

    for idx, name in enumerate(names):
        path = datasets[name]
        xs, ys, stds, runs = [], [], [], []
        if os.path.exists(path):
            results = load_results_list(path)
            xs, ys, stds, runs = extract_n_mlps_and_accuracy_with_std(results)
        r, c = positions[idx]
        color = colors[idx % len(colors)]
        fill_color = hex_to_rgba(color, alpha=0.18)

        if not xs:
            fig.add_annotation(text=f"No data for {name}", row=r, col=c, showarrow=False,
                               x=0.5, y=0.5, xref=f"x{idx+1} domain", yref=f"y{idx+1} domain")
            continue

        plot_stds = [compute_plot_std(s, n) for s, n in zip(stds, runs)]
        upper = [m + s for m, s in zip(ys, plot_stds)]
        lower = [m - s for m, s in zip(ys, plot_stds)]

        # add shaded region
        fig.add_trace(
            go.Scatter(x=xs, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=r, col=c
        )
        fig.add_trace(
            go.Scatter(x=xs, y=lower, mode="lines", fill="tonexty", fillcolor=fill_color,
                       line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=r, col=c
        )

        # main trace
        # include original std and plotted std in hover via customdata
        custom = [{"orig_std": o, "plot_std": p, "num_runs": n} for o,p,n in zip(stds, plot_stds, runs)]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                marker=dict(color=color, size=8),
                line=dict(color=color, width=2),
                name=name,
                customdata=custom,
                hovertemplate="n_mlps=%{x}<br>Accuracy=%{y:.4f}<br>std_shown=%{customdata[plot_std]:.4f}<extra></extra>"
            ),
            row=r, col=c
        )

        # compute padding so first/last visible
        min_x, max_x = min(xs), max(xs)
        pad = 0.5
        range_min = max(1.5, min_x - pad)
        range_max = min(16.5, max_x + pad)

        fig.update_xaxes(title_text="n_mlps", row=r, col=c,
                         tickmode="array", tickvals=tick_vals, ticktext=[str(v) for v in tick_vals],
                         range=[range_min, range_max])
        fig.update_yaxes(title_text="Accuracy", row=r, col=c)

    fig.update_layout(
        title="n_mlps vs Accuracy (across datasets)",
        height=900, width=1200,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=70, r=30, t=100, b=80)
    )
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-html", default=OUT_HTML)
    parser.add_argument("--out-png", default=OUT_PNG)
    parser.add_argument("--data", nargs="*", default=None, help="override dataset paths: name=/path")
    args = parser.parse_args()

    datasets = DATA_FILES.copy()
    if args.data:
        for item in args.data:
            if "=" in item:
                k, v = item.split("=", 1)
                datasets[k] = v

    fig = build_figure(datasets, COLORS)
    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    fig.write_html(args.out_html, include_plotlyjs="cdn")
    print(f"Saved interactive html: {args.out_html}")
    try:
        fig.write_image(args.out_png, scale=2)
        print(f"Saved png: {args.out_png}")
    except Exception as e:
        print("PNG save skipped (kaleido may be missing):", e)

if __name__ == "__main__":
    main()
# ...existing code...