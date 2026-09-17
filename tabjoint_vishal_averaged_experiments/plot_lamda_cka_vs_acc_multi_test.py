# ...existing code...
import json
import os
import argparse
from typing import List, Tuple, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# default input files (from your workspace)
DATA_FILES = {
    "wine": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_wine.json",
    "adult": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_adult.json",
    "titanic": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_titanic.json",
    "california_housing": "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_califonria_housing.json",
}

OUT_HTML = "/media/homes/hamza/results/best_val_vs_lambda_cka_multi_test.html"
OUT_PNG = "/media/homes/hamza/results/best_val_vs_lambda_cka_multi_test.png"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  # distinct colors for the four panels

def load_results_list(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    return [data]

def extract_lambda_and_test_mean(results: List[dict]) -> Tuple[List[float], List[float]]:
    """
    Extract (lambda_cka, test_mean) pairs.
    Uses 'test_mean' from scores, falls back to 'test' if missing.
    Std/dev ignored.
    """
    pairs = []
    for r in results:
        lam = None
        if "lambda_cka" in r:
            lam = r["lambda_cka"]
        else:
            for c in ("hparams", "config", "params"):
                if c in r and isinstance(r[c], dict) and "lambda_cka" in r[c]:
                    lam = r[c]["lambda_cka"]
                    break

        test_mean = None
        if "scores" in r and isinstance(r["scores"], dict):
            s = r["scores"]
            test_mean = s.get("test_mean", s.get("test", None))
        else:
            test_mean = r.get("test_mean", r.get("test", None))

        if lam is None or test_mean is None:
            continue
        try:
            pairs.append((float(lam), float(test_mean)))
        except Exception:
            continue

    if not pairs:
        return [], []
    pairs.sort(key=lambda x: x[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return xs, ys

def build_figure(datasets: Dict[str, str], colors: List[str]) -> go.Figure:
    names = list(datasets.keys())
    fig = make_subplots(rows=2, cols=2, subplot_titles=names, shared_xaxes=False)
    subplot_positions = [(1,1), (1,2), (2,1), (2,2)]
    for idx, name in enumerate(names):
        path = datasets[name]
        xs, ys = [], []
        if os.path.exists(path):
            results = load_results_list(path)
            xs, ys = extract_lambda_and_test_mean(results)
        color = colors[idx % len(colors)]
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
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name=name,
                marker=dict(color=color, size=8),
                line=dict(color=color, width=2),
                hovertemplate="lambda_cka=%{x}<br>Test Accuracy=%{y:.4f}<extra></extra>"
            ),
            row=r, col=c
        )
        fig.update_xaxes(title_text="lambda_cka", row=r, col=c)
        fig.update_yaxes(title_text="Test Accuracy", row=r, col=c)
    fig.update_layout(
        title_text="Test Accuracy (test_mean) vs lambda_cka (4 datasets)",
        height=900, width=1200,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
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
# ...existing code...