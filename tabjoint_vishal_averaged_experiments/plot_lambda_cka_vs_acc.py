# ...existing code...
import json
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Tuple

RESULTS_JSON = "/media/homes/hamza/results/lambda_cka_averaged_exp/tabjoint_search_results_titanic.json"
SAVE_PNG = "/media/homes/hamza/results/lambda_cka_averaged_exp/lambda_cka_vs_test_accuracy_titanic.png"
# ...existing code...

def load_results(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    # accept either list-of-runs or dict with "results" key
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    # fallback: wrap single dict
    return [data]

def extract_lambda_cka_and_test_accuracy(results: List[dict]) -> Tuple[List[float], List[float]]:
    """
    Extract pairs (lambda_cka, test_accuracy) from results.
    Requires lambda_cka (top-level or inside 'hparams'/'config'/'params')
    and the 'test' metric (top-level or inside 'scores'/'metrics'/'results'/'scores').
    """
    pairs = []
    for r in results:
        # find lambda_cka (top-level or nested)
        lambda_cka = None
        if "lambda_cka" in r:
            lambda_cka = r["lambda_cka"]
        else:
            for container in ("hparams", "config", "params"):
                if container in r and isinstance(r[container], dict) and "lambda_cka" in r[container]:
                    lambda_cka = r[container]["lambda_cka"]
                    break

        # find test accuracy explicitly
        test_acc = None
        if "best_val" in r:
            test_acc = r["best_val"]
        else:
            for container in ("scores", "metrics", "results", "scores"):
                if container in r and isinstance(r[container], dict) and "best_val" in r[container]:
                    test_acc = r[container]["best_val"]
                    break

        if lambda_cka is None or test_acc is None:
            # skip runs missing either field
            continue

        try:
            pairs.append((float(lambda_cka), float(test_acc)))
        except Exception:
            continue

    if not pairs:
        return [], []

    pairs.sort(key=lambda x: x[0])
    xs, ys = zip(*pairs)
    return list(xs), list(ys)

def plot_lambda_cka_vs_accuracy(lambda_cka_vals, accuracies, save_path: str = None, show: bool = True):
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_cka_vals, accuracies, marker='o', linestyle='-')
    plt.xlabel("lambda_cka")
    plt.ylabel("Test Accuracy")
    plt.title("lambda_cka vs Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    if show:
        try:
            plt.show()
        except Exception:
            plt.close()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "-r", default=RESULTS_JSON, help="Path to results json")
    parser.add_argument("--out", "-o", default=SAVE_PNG, help="Where to save the plot png")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        return
    results = load_results(args.results)
    lambda_cka_vals, accuracies = extract_lambda_cka_and_test_accuracy(results)
    if not lambda_cka_vals or not accuracies:
        print("No valid lambda_cka and test accuracy data found in results.")
        return
    plot_lambda_cka_vs_accuracy(lambda_cka_vals, accuracies, save_path=args.out, show=not args.no_show)

if __name__ == "__main__":
    main()
# ...existing code...