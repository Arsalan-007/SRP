import json
import os
import argparse
from typing import List, Tuple, Dict, Optional
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# default input files (from your workspace)
DATA_FILES = {
    "wine": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_wine.json",
    "adult": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_adult.json",
    "churn_modelling": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_Churn_Modeling.json",
    "california_housing": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_califonria_housing.json",
}

OUT_HTML = "/media/homes/hamza/results/best_val_vs_lambda_cka_multi.html"
OUT_PNG = "/media/homes/hamza/results/best_val_vs_lambda_cka_multi.png"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  # distinct colors for the four panels

def load_results_list(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    return [data]

def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
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

def extract_lambda_best_and_std(results: List[dict]) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    Extract (lambda_cka, best_val_mean, best_val_std, num_runs) records.
    If std or num_runs missing, std -> 0.0, num_runs -> 1
    """
    records = []
    for r in results:
        lam = None
        if "lambda_cka" in r:
            lam = r["lambda_cka"]
        else:
            for c in ("hparams", "config", "params"):
                if c in r and isinstance(r[c], dict) and "lambda_cka" in r[c]:
                    lam = r[c]["lambda_cka"]
                    break

        best_mean = None
        best_std = None
        num_runs = None
        if "scores" in r and isinstance(r["scores"], dict):
            s = r["scores"]
            best_mean = s.get("best_val_mean", s.get("best_val"))
            best_std = s.get("best_val_std", 0.0)
            num_runs = s.get("num_runs", None)
        else:
            best_mean = r.get("best_val_mean", r.get("best_val"))
            best_std = r.get("best_val_std", 0.0)
            num_runs = r.get("num_runs", None)

        if lam is None or best_mean is None:
            continue
        try:
            records.append((float(lam), float(best_mean), float(best_std or 0.0), int(num_runs) if num_runs is not None else 1))
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
    Reduce plotted uncertainty by converting to standard error (std / sqrt(n)) when possible.
    Fallback: if num_runs <= 1, shrink by factor 0.5 to visually reduce shading.
    This only affects plotting; data not modified.
    """
    if std is None:
        return 0.0
    try:
        if num_runs and num_runs > 1:
            return std / math.sqrt(num_runs)
    except Exception:
        pass
    # fallback shrink
    return std * 0.5

def build_figure(datasets: Dict[str, str], colors: List[str]) -> go.Figure:
    names = list(datasets.keys())
    fig = make_subplots(rows=2, cols=2, subplot_titles=names, shared_xaxes=False)
    subplot_positions = [(1,1), (1,2), (2,1), (2,2)]
    for idx, name in enumerate(names):
        path = datasets[name]
        xs, ys, stds, runs = [], [], [], []
        if os.path.exists(path):
            results = load_results_list(path)
            xs, ys, stds, runs = extract_lambda_best_and_std(results)
        color = colors[idx % len(colors)]
        fill_color = hex_to_rgba(color, alpha=0.18)
        r, c = subplot_positions[idx]
        if not xs:
            fig.add_annotation(
                text=f"No data found for {name}",
                xref=f"x{idx+1} domain",
                yref=f"y{idx+1} domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=r, col=c
            )
            continue

        # compute reduced plotting std (standard error when num_runs available)
        plot_stds = [compute_plot_std(s, n) for s, n in zip(stds, runs)]

        # upper and lower bounds using reduced stds
        upper = [m + s for m, s in zip(ys, plot_stds)]
        lower = [m - s for m, s in zip(ys, plot_stds)]

        # add upper trace (invisible) then lower trace with fill to previous (upper) to create shaded region
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r, col=c
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor=fill_color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r, col=c
        )

        # main mean trace on top; include original std and plotted (reduced) std in hover
        custom = [{"orig_std": orig, "plot_std": p} for orig, p in zip(stds, plot_stds)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                marker=dict(color=color, size=8),
                line=dict(color=color, width=2),
                hovertemplate="lambda_cka=%{x}<br>Accuracy=%{y:.4f}<br>std_shown=%{customdata[plot_std]:.4f}<extra></extra>",
                customdata=custom,
            ),
            row=r, col=c
        )

        # add padding so first/last points visible
        min_x, max_x = min(xs), max(xs)
        pad = (max_x - min_x) * 0.05 if max_x > min_x else 0.05
        fig.update_xaxes(title_text="lambda_cka", row=r, col=c, range=[min_x - pad, max_x + pad])
        fig.update_yaxes(title_text="Accuracy", row=r, col=c)

    fig.update_layout(
        title_text="Accuracy vs lambda_cka (across datasets)",
        height=900, width=1200,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=70, r=30, t=100, b=70),
    )
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-html", default=OUT_HTML, help="Path to save interactive html")
    parser.add_argument("--out-png", default=OUT_PNG, help="Path to save static png (requires kaleido)")
    parser.add_argument("--data", nargs="*", default=None, help="Optional dataset=path pairs to override defaults")
    args = parser.parse_args()

    datasets = DATA_FILES.copy()
    if args.data:
        for item in args.data:
            if "=" in item:
                k, v = item.split("=", 1)
                datasets[k] = v

    fig = build_figure(datasets, COLORS)
    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    fig.write_html(args.out_html, include_plotlyjs="cdn")
    print(f"Saved interactive plot to {args.out_html}")

    try:
        fig.write_image(args.out_png, scale=2)
        print(f"Saved PNG to {args.out_png}")
    except Exception as e:
        print("Could not save PNG (kaleido missing?). PNG save skipped. Error:", e)

if __name__ == "__main__":
    main()