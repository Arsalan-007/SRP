
import os
import argparse
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, CSV_PATH, TARGET_COL
from .preprocessing import TabularPreprocessor, PrepConfig, build_label_mapping, apply_mapping
from .hparams import GRID, STATIC, RESULTS_JSON, BEST_JSON, PRIMARY_METRIC, EXPERIMENT, NUM_RUNS
from .grid_search import run_grid_search
import json
from .hparams import GRID, STATIC, RESULTS_JSON, BEST_JSON, PRIMARY_METRIC, EXPERIMENT, NUM_RUNS
import json, os, csv

# ...existing code...
def _set_visible_devices(gpus: str | None):
    """
    Set CUDA_VISIBLE_DEVICES from the --gpus argument.

    - Accepts None or "" -> do nothing.
    - Accepts comma-separated numeric physical GPU ids, e.g. "4" or "3,4".
    - Sets CUDA_DEVICE_ORDER=PCI_BUS_ID to ensure predictable mapping.
    - Sets OMP_NUM_THREADS=1 if not already set.
    """
    if gpus is None:
        return
    g = gpus.strip()
    if g == "":
        return

    # sanitize tokens and ensure they are numeric (physical GPU ids)
    tokens = [t.strip() for t in g.split(",") if t.strip()]
    if not tokens:
        return
    for t in tokens:
        if not t.isdigit():
            # invalid format: do not change environment and warn
            print(f"[gs_main] warning: invalid --gpus token '{t}' (expected digits). No change to CUDA_VISIBLE_DEVICES.")
            return

    devices = ",".join(tokens)
    # ensure consistent device ordering
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # quick debug/info
    print(f"[gs_main] CUDA_VISIBLE_DEVICES set to '{devices}' (requested '{gpus}')")
# ...existing code...

# ...existing code...
def run(gpus: str | None = None):
    _set_visible_devices(gpus)

    import os
    import torch

    # Env / mapping info
    print("DEBUG: CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("DEBUG: CUDA_DEVICE_ORDER =", os.environ.get("CUDA_DEVICE_ORDER"))
    print("DEBUG: torch.cuda.is_available() =", torch.cuda.is_available())

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("DEBUG: torch.cuda.device_count() =", count)
        for i in range(count):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = "<unknown>"
            print(f"DEBUG: visible device index {i} -> name: {name}")
        cur = torch.cuda.current_device()
        print("DEBUG: torch.cuda.current_device() =", cur)
        print("DEBUG: torch.cuda.device_name(current) =", torch.cuda.get_device_name(cur))
# ...existing code...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    print(f"[gs_main] device={device} | visible_gpus='{os.environ.get('CUDA_VISIBLE_DEVICES','all')}'")
    print(f"[gs_main] EXPERIMENT={EXPERIMENT} | NUM_RUNS={NUM_RUNS}\n")

    df = pd.read_csv(CSV_PATH)
    df = df.iloc[:-1]
    if 'PassengerId' in df.columns:
        df = df.drop('PassengerId', axis=1)

    df = df.dropna(subset=[TARGET_COL])

    X_raw = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]
    y_raw = df[TARGET_COL].values
    X_raw = df.drop(columns=[TARGET_COL])

    is_reg = (y_raw.dtype.kind in {"f"} and np.unique(y_raw).size > 10)

    # Dictionary to aggregate results by the experiment parameter
    aggregated = {}
    print("NUM_RUNS:", NUM_RUNS)
    for run_idx in range(NUM_RUNS):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx+1}/{NUM_RUNS}")
        print(f"{'='*70}")
        
        run_seed = SEED + run_idx * 1000
        
        if is_reg:
            X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=run_seed)
            X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(X_tr, y_tr_raw, test_size=0.2, random_state=run_seed)
        else:
            X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=run_seed, stratify=y_raw)
            X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(X_tr, y_tr_raw, test_size=0.2, random_state=run_seed, stratify=y_tr_raw)

        prep = TabularPreprocessor(
            PrepConfig(
                impute_num="mean", impute_cat="most_frequent",
                scale="standard", noise_std=0.0,
                cardinality_threshold=50, rare_threshold_pct=0.05,
                cat_encoding="onehot", other_token="__OTHER__", missing_token="__MISSING__"
            )
        ).fit(X_tr)

        X_train_t = prep.transform(X_tr)
        X_val_t   = prep.transform(X_va)
        X_test_t  = prep.transform(X_te)

        info = build_label_mapping(y_tr_raw)
        task, out_dim = info["task"], info["out_dim"]
        y_train = apply_mapping(y_tr_raw, info)
        y_val   = apply_mapping(y_va_raw, info)
        y_test  = apply_mapping(y_te_raw, info)

        in_dim = X_train_t.shape[1]
        print(task, out_dim, in_dim, X_train_t.shape, X_val_t.shape, X_test_t.shape)

        best, trials = run_grid_search(
            X_train_t, y_train, X_val_t, y_val, X_test_t, y_test,
            in_dim=in_dim, task=task, out_dim=out_dim, device=device,
            grid=GRID, static=STATIC,
            results_path=RESULTS_JSON, best_path=BEST_JSON,
            primary_metric=PRIMARY_METRIC
        )
        
        
        for trial in trials:
            param_value = trial["hparams"].get(EXPERIMENT)
            if param_value is None:
                continue
            if param_value not in aggregated:
                aggregated[param_value] = {
                    "best_vals": [], "test_vals": [],
                    "sim_cka_vals": [], "sim_cos_vals": []
                }

            # scores
            scores = trial.get("scores", {}) or {}
            bv = scores.get("best_val"); tv = scores.get("test")
            if bv is not None: aggregated[param_value]["best_vals"].append(bv)
            if tv is not None: aggregated[param_value]["test_vals"].append(tv)

            # similarities (trial may have them at root)
            sim_cka = trial.get("mean_similarity_cka", None)
            sim_cos = trial.get("mean_similarity_cosine", None)
            if sim_cka is None:
                sim_cka = scores.get("mean_similarity_cka", None)
            if sim_cos is None:
                sim_cos = scores.get("mean_similarity_cosine", None)
            if sim_cka is not None:
                aggregated[param_value]["sim_cka_vals"].append(sim_cka)
            if sim_cos is not None:
                aggregated[param_value]["sim_cos_vals"].append(sim_cos)

    
    # Compute averages and write aggregated results
        # Build aggregated_trials
    aggregated_trials = []
    for param_value in sorted(aggregated.keys()):
        best_list = aggregated[param_value]["best_vals"]
        test_list = aggregated[param_value]["test_vals"]
        sim_cka_list = aggregated[param_value]["sim_cka_vals"]
        sim_cos_list = aggregated[param_value]["sim_cos_vals"]

        mean_best = float(np.mean(best_list)) if best_list else None
        std_best  = float(np.std(best_list))  if best_list else None
        mean_test = float(np.mean(test_list)) if test_list else None
        std_test  = float(np.std(test_list))  if test_list else None

        mean_sim_cka = float(np.mean(sim_cka_list)) if sim_cka_list else None
        std_sim_cka  = float(np.std(sim_cka_list))  if sim_cka_list else None
        mean_sim_cos = float(np.mean(sim_cos_list)) if sim_cos_list else None
        std_sim_cos  = float(np.std(sim_cos_list))  if sim_cos_list else None

        aggregated_trials.append({
            "hparams": {EXPERIMENT: param_value},
            "scores": {
                "best_val_mean": mean_best,
                "best_val_std": std_best,
                "test_mean": mean_test,
                "test_std": std_test,
                "mean_similarity_cka": mean_sim_cka,
                "std_similarity_cka": std_sim_cka,
                "mean_similarity_cosine": mean_sim_cos,
                "std_similarity_cosine": std_sim_cos,
                "num_runs": NUM_RUNS,
                "best_vals_all_runs": best_list,
                "test_all_runs": test_list,
                "sim_cka_all_runs": sim_cka_list,
                "sim_cos_all_runs": sim_cos_list
            }
        })

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(aggregated_trials, f, indent=2)
    print(f"Aggregated JSON saved to {RESULTS_JSON}")
    csv_path = RESULTS_JSON.replace(".json", ".csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    header = [
        "experiment", "param_name", "param_value",
        "mean_similarity_cka", "std_similarity_cka",
        "mean_similarity_cosine", "std_similarity_cosine",
        "accuracy_mean", "accuracy_std",
        "num_runs"
    ]
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        for entry in aggregated_trials:
            p = entry["hparams"].get(EXPERIMENT)
            s = entry["scores"]
            row = [
                EXPERIMENT, EXPERIMENT, p,
                s.get("mean_similarity_cka"),
                s.get("std_similarity_cka"),
                s.get("mean_similarity_cosine"),
                s.get("std_similarity_cosine"),
                s.get("test_mean"),
                s.get("test_std"),
                s.get("num_runs")
            ]
            writer.writerow(row)
    print(f"Appended aggregated rows to {csv_path}")
    print(f"{'='*70}\n")
    return aggregated_trials


def cli_main():
    parser = argparse.ArgumentParser(description="Grid search launcher (CUDA-friendly).")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0' or '0,1'")
    args = parser.parse_args()
    run(gpus=args.gpus)

if __name__ == "__main__":
    cli_main()
