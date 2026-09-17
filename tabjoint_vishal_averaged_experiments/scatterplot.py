import os
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_FILES: Dict[str, str] = {
    "wine": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_wine.csv",
    "adult": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_adult.csv",
    "churn_modelling": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_Churn_Modeling.csv",
    "california_housing": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_califonria_housing.csv",
}

OUT_HTML = "/media/homes/hamza/results/cka_vs_acc_multi_scatter.html"
OUT_PNG = "/media/homes/hamza/results/cka_vs_acc_multi_scatter.png"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def prepare_all(datasets: Dict[str, str]) -> Tuple[Dict[str, pd.DataFrame], float, float]:
    data: Dict[str, pd.DataFrame] = {}
    global_x_min = float("inf")
    global_x_max = float("-inf")
    for name, path in datasets.items():
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        if "mean_similarity_cka" not in df.columns or "accuracy_mean" not in df.columns:
            continue
        df = df.dropna(subset=["mean_similarity_cka", "accuracy_mean"])
        if df.empty:
            continue
        xmin = df["mean_similarity_cka"].min()
        xmax = df["mean_similarity_cka"].max()
        global_x_min = min(global_x_min, xmin)
        global_x_max = max(global_x_max, xmax)
        data[name] = df
    if global_x_min == float("inf"):
        global_x_min, global_x_max = 0.0, 1.0
    # add small padding
    pad = (global_x_max - global_x_min) * 0.05 if global_x_max > global_x_min else 0.05
    return data, global_x_min - pad, global_x_max + pad

def build_figure(data: Dict[str, pd.DataFrame], x_range: Tuple[float, float], colors: List[str]) -> go.Figure:
    names = list(data.keys())
    rows = 2
    cols = 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, shared_xaxes=False, shared_yaxes=False)
    positions = [(1,1), (1,2), (2,1), (2,2)]
    for idx, name in enumerate(names):
        r, c = positions[idx]
        df = data[name]
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df["mean_similarity_cka"],
                y=df["accuracy_mean"],
                mode="markers",
                marker=dict(size=10, color=color, opacity=0.9),
                name=name,
                hovertemplate="mean_similarity_cka=%{x:.4f}<br>accuracy_mean=%{y:.4f}<extra></extra>"
            ),
            row=r, col=c
        )
        fig.update_xaxes(title_text="mean_similarity_cka", row=r, col=c, range=x_range)
        fig.update_yaxes(title_text="accuracy_mean", row=r, col=c, autorange=True)
    fig.update_layout(
        title="mean_similarity_cka vs accuracy_mean (multiple datasets)",
        height=900, width=1200,
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=60, r=30, t=90, b=60)
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

    data, xmin, xmax = prepare_all(datasets)
    if not data:
        print("No valid datasets found or required columns missing.")
        return

    fig = build_figure(data, (xmin, xmax), COLORS)
    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    fig.write_html(args.out_html, include_plotlyjs="cdn")
    print(f"Saved interactive HTML: {args.out_html}")

    try:
        fig.write_image(args.out_png, scale=2)
        print(f"Saved PNG: {args.out_png}")
    except Exception as e:
        print("PNG save skipped (kaleido may be missing):", e)

if __name__ == "__main__":
    main()