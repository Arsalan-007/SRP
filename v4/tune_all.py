#!/usr/bin/env python3
"""
tune_all.py  —  SRP v4

Per-dataset Optuna tuning across all 8 benchmarks: priority 3 on the interim deck's
Next Steps slide, "What is our architecture's true ceiling?".

Every notebook so far fixed the architecture and varied one thing. This does the opposite:
each dataset gets its own full search, so the answer is the best this architecture can do
rather than the best the step-3 defaults can do.

Two deliberate departures from the search notebooks 09-19 used, both because the question
is about the CEILING:

  * wide=True        hidden_dim 8-512 (was 8-64), embed_dim up to 256 (was 64), depth 1-3.
                     Notebook 20 showed capacity is the binding constraint on Covertype
                     (+1.73% going from hidden 16 to 256), so searching under the old cap
                     would report the ceiling of the cap, not of the architecture.
  * search_shared    Optuna picks per dataset between independent experts and TabM-style
                     shared weights with rank-1 adapters. Notebook 20 found independent
                     wins on CA/HI and shared wins on CO, so there is no single right
                     answer to hard-code — which is the whole argument for tuning per
                     dataset.

The default arm is `conf_uniform` (predict-then-average), not `meanpool`: notebook 17
measured the pooling point as worth 0.0194 RMSE on California at zero parameter cost, so
tuning `meanpool` would be tuning the weaker architecture.

Resumable. Each (dataset, arm) writes into results/tuning/<key>/<arm>/ and is skipped if
its summary.json is already there, so an interrupted run is restarted with the same
command. Optuna's journal file also resumes a partially finished search.

Usage (from ~/SRP/v4 on gpu01 — see run_tuning.sh for the nohup form):

    CUDA_VISIBLE_DEVICES=3 python3 tune_all.py                    # all 8, conf_uniform
    CUDA_VISIBLE_DEVICES=3 python3 tune_all.py --datasets CA HI   # a subset
    CUDA_VISIBLE_DEVICES=3 python3 tune_all.py --arms conf_uniform meanpool
    python3 tune_all.py --smoke                                   # 2 min end-to-end check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import benchmarks
import benchmarks_rtdl
from results import save_report
from training import get_device
from tuning import (create_study, load_study, optuna_worker, run_config, tune_xgb,
                    make_xgb, xgb_metrics)

__all__ = ["BUDGETS", "LOADERS", "tune_one", "main"]

LOADERS = {**{k: benchmarks_rtdl for k in benchmarks_rtdl.BENCHMARKS},
           **{k: benchmarks for k in benchmarks.BENCHMARKS}}

# Cheapest first, so a run that is killed early still leaves usable results.
# trials / trial_epochs are tiered by cost = trials x trial_epochs x train_rows x n_features
# (each feature gets its own PLR embedding, so cost is linear in n_features). Santander,
# YearPrediction and Covertype are ~70% of the total; Santander alone is ~37%, the single
# most expensive dataset, because of its 200 features. Those get fewer, shorter trials and
# lean on the MedianPruner.
# epochs/patience apply to BOTH the search trials and the final re-train. The first run of
# this script used short trials (40-80 epochs) and a 200-epoch final, and that mismatch cost
# accuracy on HI, JA and YE: a config that wins an 80-epoch trial is not the config that wins
# a 200-epoch run, and Optuna duly picked fast-converging small models (Jannis: K=2, no
# embeddings, 16k params). Rank trials under the budget the final model actually gets.
BUDGETS = {
    "CA": dict(trials=120, epochs=200, patience=20),
    "AD": dict(trials=120, epochs=200, patience=20),
    "HI": dict(trials=100, epochs=200, patience=20),
    "JA": dict(trials=100, epochs=200, patience=20),
    "SA": dict(trials=60,  epochs=200, patience=20),
    "FB": dict(trials=60,  epochs=200, patience=20),
    "CO": dict(trials=120, epochs=200, patience=20),
    "YE": dict(trials=40,  epochs=200, patience=20),
}
DATASET_ORDER = ["CA", "AD", "HI", "JA", "SA", "FB", "CO", "YE"]

# The widened space reaches ~22M-parameter models (K=64, hidden=512, independent) whose
# trials cost hours. Everything this project has run is under 400k, so 4M is already far
# past any configuration we expect to win, while keeping trials comparably priced.
MAX_PARAMS = 4_000_000


def _log(msg: str, path: Path = None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _score(metrics: Dict, task: str, y_std: float) -> float:
    """The dataset's own metric, RMSE rescaled to ORIGINAL units like cli.py:104 and the
    published tables. Notebook 11 skipped this rescaling; its numbers are not comparable
    to published ones. Do not remove it."""
    return metrics["test_rmse"] * y_std if task == "regression" else metrics["test_acc"]


def tune_one(key: str, arm: str, out_root: Path, n_jobs: int, threads: int,
             final_seeds: int, final_epochs: int, final_patience: int,
             budget: Dict, wide: bool, search_shared: bool, log_path: Path) -> Dict:
    """Search + final re-train for one (dataset, arm). Returns the summary dict."""
    out_dir = out_root / key / arm
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        _log(f"  {key}/{arm}: already done, skipping (delete {summary_path} to redo)", log_path)
        return json.load(open(summary_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    # The CLI overrides are escape hatches; 0 means "use the dataset's budget".
    epochs = final_epochs or budget["epochs"]
    patience = final_patience or budget["patience"]
    loader = LOADERS[key]
    data = loader.load_benchmark(key)
    arrays = data.arrays()
    hib = loader.higher_is_better(key)
    metric = "rmse" if data.task == "regression" else "acc"
    y_std = getattr(data, "y_std", 1.0)

    study_name = f"{key}_{arm}"
    storage = str(out_dir / "study.journal")
    study = create_study(study_name, storage)
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    todo = max(0, budget["trials"] - done)
    _log(f"  {key}/{arm}: {data.label} | {len(arrays['ytr']):,} train rows | "
         f"{data.n_features} features | {data.task} | {done} trials done, {todo} to go", log_path)

    t0 = time.time()
    if todo:
        per_worker = int(np.ceil(todo / n_jobs))
        Parallel(n_jobs=n_jobs)(
            delayed(optuna_worker)(arm, study_name, storage, per_worker, 100 + w, arrays,
                                   task=data.task, output_dim=data.output_dim, threads=threads,
                                   trial_epochs=epochs, trial_patience=patience,
                                   project_dir=PROJECT_DIR, wide=wide,
                                   search_shared=search_shared, max_params=MAX_PARAMS)
            for w in range(n_jobs))
    search_min = (time.time() - t0) / 60
    study = load_study(study_name, storage)
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    _log(f"  {key}/{arm}: search {search_min:.1f} min | {states} | "
         f"best val loss {study.best_value:.4f}", log_path)

    best = dict(study.best_params)
    json.dump(best, open(out_dir / "best_params.json", "w"), indent=2)

    t0 = time.time()
    rows = Parallel(n_jobs=min(n_jobs, final_seeds))(
        delayed(run_config)(arm, best, arrays, seed, task=data.task,
                            output_dim=data.output_dim, epochs=epochs,
                            patience=patience, threads=threads)
        for seed in range(final_seeds))
    final_min = (time.time() - t0) / 60

    scores = []
    for r in rows:
        s = _score(r, data.task, y_std)
        scores.append(s)
        save_report(out_dir / f"seed{r['seed']}",
                    dataset=key, label=data.label, arm=arm, seed=r["seed"],
                    source="reproduced", source_ref="tune_all.py (Optuna-tuned)",
                    config={"best_params": best, "wide": wide, "search_shared": search_shared,
                            "trials": budget["trials"], "epochs": epochs, "patience": patience},
                    n_parameters=r["n_params"],
                    metrics={"test": {metric: s}},
                    champion_metric=f"test_{metric}", champion_value=s,
                    higher_is_better=hib, time_seconds=r["seconds"], checkpoint=None)

    scores = np.array(scores)
    summary = {"dataset": key, "label": data.label, "arm": arm, "task": data.task,
               "metric": metric, "higher_is_better": hib,
               "mean": float(scores.mean()), "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
               "per_seed": [float(s) for s in scores],
               "n_parameters": int(rows[0]["n_params"]),
               "best_params": best, "n_trials": len(study.trials),
               "best_val_loss": float(study.best_value),
               "search_minutes": search_min, "final_minutes": final_min}
    json.dump(summary, open(summary_path, "w"), indent=2)
    _log(f"  {key}/{arm}: DONE {metric} {scores.mean():.4f} +/- {summary['std']:.4f} "
         f"({summary['n_parameters']:,} params, final {final_min:.1f} min)", log_path)
    return summary


def tune_xgb_one(key: str, out_root: Path, n_trials: int, final_seeds: int,
                 log_path: Path) -> Dict:
    """XGBoost with the SAME trial count, so neither side of the comparison gets more
    search effort than the other."""
    out_dir = out_root / key / "xgboost"
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        _log(f"  {key}/xgboost: already done, skipping", log_path)
        return json.load(open(summary_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = LOADERS[key]
    data = loader.load_benchmark(key)
    arrays = data.arrays()
    hib = loader.higher_is_better(key)
    metric = "rmse" if data.task == "regression" else "acc"
    y_std = getattr(data, "y_std", 1.0)
    ytr = arrays["ytr"] if data.task == "regression" else arrays["ytr"].astype(int)

    t0 = time.time()
    study = tune_xgb(arrays, data.task, n_trials=n_trials)
    scores = []
    for seed in range(final_seeds):
        m = make_xgb(study.best_params, data.task, seed).fit(arrays["Xtr"], ytr)
        mt = xgb_metrics(m, arrays["Xte"], arrays["yte"], data.task)
        scores.append(mt["rmse"] * y_std if data.task == "regression" else mt["acc"])
    scores = np.array(scores)
    summary = {"dataset": key, "label": data.label, "arm": "xgboost", "task": data.task,
               "metric": metric, "higher_is_better": hib,
               "mean": float(scores.mean()),
               "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
               "per_seed": [float(s) for s in scores], "n_parameters": 0,
               "best_params": study.best_params, "n_trials": len(study.trials),
               "best_val_loss": float(study.best_value),
               "search_minutes": (time.time() - t0) / 60, "final_minutes": 0.0}
    json.dump(summary, open(summary_path, "w"), indent=2)
    _log(f"  {key}/xgboost: DONE {metric} {scores.mean():.4f} +/- {summary['std']:.4f}", log_path)
    return summary


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASET_ORDER,
                    choices=sorted(LOADERS), help="default: all 8, cheapest first")
    ap.add_argument("--arms", nargs="+", default=["conf_uniform"])
    ap.add_argument("--xgboost", action="store_true",
                    help="also tune XGBoost per dataset with the same trial count")
    ap.add_argument("--out", default="results/tuning")
    ap.add_argument("--jobs", type=int, default=0, help="0 = 3 on GPU, cpu_count/2 on CPU")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, default=5)
    ap.add_argument("--final-epochs", type=int, default=0,
                    help="0 = use the dataset budget (and keep search/final identical)")
    ap.add_argument("--final-patience", type=int, default=0,
                    help="0 = use the dataset budget")
    ap.add_argument("--trials", type=int, default=0, help="override the per-dataset budget")
    ap.add_argument("--narrow", action="store_true",
                    help="use the old hidden<=64 space (reproduces notebooks 09-19)")
    ap.add_argument("--no-shared", action="store_true",
                    help="do not let Optuna choose TabM-style weight sharing")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check: 2 datasets, 4 trials, 3 epochs, 1 seed")
    a = ap.parse_args(argv)

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # pid in the name: two shards launched in the same second would otherwise share
    # one log file and interleave into it.
    log_path = out_root / f"tune_all_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"

    device = get_device()
    n_jobs = a.jobs or (3 if device.type == "cuda" else max(1, (os.cpu_count() or 2) // a.threads))
    datasets, arms = list(a.datasets), list(a.arms)
    budgets = {k: dict(BUDGETS[k]) for k in datasets}
    final_seeds, final_epochs, final_patience = a.final_seeds, a.final_epochs, a.final_patience

    if a.smoke:
        datasets, arms = ["CA", "AD"], arms[:1]
        budgets = {k: dict(trials=4, epochs=3, patience=2) for k in datasets}
        final_seeds, final_epochs, final_patience = 1, 3, 2
        out_root = out_root / "_smoke"
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / "smoke.log"
    if a.trials:
        for k in budgets:
            budgets[k]["trials"] = a.trials

    wide, search_shared = not a.narrow, not a.no_shared
    total = sum(budgets[k]["trials"] for k in datasets) * len(arms)
    _log(f"device {device} | {n_jobs} workers x {a.threads} threads | "
         f"{len(datasets)} datasets x {len(arms)} arms = {total} trials", log_path)
    ep = {k: final_epochs or budgets[k]["epochs"] for k in datasets}
    _log(f"space: {'WIDE (hidden<=512)' if wide else 'narrow (hidden<=64)'} | "
         f"shared weights {'searched' if search_shared else 'off'} | "
         f"search and final both {min(ep.values())}-{max(ep.values())} epochs | "
         f"final {final_seeds} seeds | log {log_path}", log_path)

    summaries = []
    for key in datasets:
        _log(f"=== {key} ===", log_path)
        for arm in arms:
            try:
                summaries.append(tune_one(key, arm, out_root, n_jobs, a.threads, final_seeds,
                                          final_epochs, final_patience, budgets[key],
                                          wide, search_shared, log_path))
            except Exception as e:                      # one dataset must not kill the run
                _log(f"  {key}/{arm}: FAILED {type(e).__name__}: {e}", log_path)
        if a.xgboost:
            try:
                summaries.append(tune_xgb_one(key, out_root, budgets[key]["trials"],
                                              final_seeds, log_path))
            except Exception as e:
                _log(f"  {key}/xgboost: FAILED {type(e).__name__}: {e}", log_path)

    if summaries:
        df = pd.DataFrame([{k: v for k, v in s.items()
                            if k not in ("best_params", "per_seed")} for s in summaries])
        csv = out_root / ("summary.csv" if len(datasets) == len(DATASET_ORDER)
                          else "summary_" + "_".join(datasets) + ".csv")
        df.to_csv(csv, index=False)
        _log(f"\nwrote {csv}", log_path)
        print(df.to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# Self-test:  python tune_all.py --selftest
# ---------------------------------------------------------------------------
def _selftest() -> int:
    import optuna
    from tuning import suggest_params, build_model
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fails = []

    def check(name, cond, extra=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")
        if not cond:
            fails.append(name)

    # 1. every budget key is a real dataset, and every dataset has a budget
    check("budgets cover exactly the 8 datasets",
          set(BUDGETS) == set(LOADERS) == set(DATASET_ORDER), f"{sorted(BUDGETS)}")

    # 1b. the search/final budget mismatch that cost accuracy on HI, JA and YE must not
    #     come back: one epochs/patience per dataset, used for both, no trial_* keys.
    check("budgets expose one epochs/patience per dataset",
          all(set(b) == {"trials", "epochs", "patience"} for b in BUDGETS.values()),
          f"{sorted(set().union(*(set(b) for b in BUDGETS.values())))}")
    import inspect
    src = inspect.getsource(tune_one)
    check("tune_one passes the same budget to search and final",
          "trial_epochs=epochs, trial_patience=patience" in src
          and "epochs=epochs" in src and "patience=patience" in src)

    # 2. the narrow space is unchanged -- notebooks 09-19 must still reproduce
    st = optuna.create_study()
    narrow = [suggest_params(t, "meanpool") for t in
              [st.ask() for _ in range(40)]]
    check("narrow space keeps hidden_dim <= 64",
          max(p["hidden_dim"] for p in narrow) <= 64)
    check("narrow space keeps embed_dim <= 64",
          max(p["embed_dim"] for p in narrow) <= 64)
    check("narrow space has no 'shared' key", all("shared" not in p for p in narrow))

    # 3. the wide space actually reaches past the old cap, and can pick sharing
    st2 = optuna.create_study()
    wide = [suggest_params(st2.ask(), "conf_uniform", wide=True, search_shared=True)
            for _ in range(60)]
    check("wide space exceeds hidden_dim 64", max(p["hidden_dim"] for p in wide) > 64)
    check("wide space exceeds embed_dim 64", max(p["embed_dim"] for p in wide) > 64)
    check("wide space offers both shared settings",
          {p["shared"] for p in wide} == {True, False})

    # 4. conf_uniform is actually buildable from a sampled config, both parameterisations
    for shared in (False, True):
        p = dict(num_learners=8, hidden_dim=128, embed_dim=32, depth=1, feature_frac=0.7,
                 dropout=0.1, lr=1e-3, weight_decay=1e-4, head_hidden=64, head_depth=1,
                 embedding_mode="periodic", d_embedding=8, n_frequencies=16, sigma=0.1,
                 shared=shared)
        m = build_model("conf_uniform", p, 8, seed=0)
        x = torch.randn(4, 8)
        out = m(x)
        check(f"conf_uniform builds+forwards with shared={shared}",
              out["pred"].shape == (4,), f"{m.num_parameters():,} params")

    # 5. the head knobs reach the model (this is what make_arm had to be extended for)
    base = dict(num_learners=4, hidden_dim=16, embed_dim=32, depth=1, feature_frac=1.0,
                dropout=0.1, lr=1e-3, weight_decay=1e-4, head_depth=1, embedding_mode="none")
    small = build_model("conf_uniform", {**base, "head_hidden": 32}, 8, 0).num_parameters()
    big = build_model("conf_uniform", {**base, "head_hidden": 512}, 8, 0).num_parameters()
    check("head_hidden changes the parameter count", big > small, f"{small:,} -> {big:,}")

    # 6. multiclass works (Covertype/Jannis need it; it was blocked until notebook 20)
    mc = build_model("conf_uniform", {**base, "head_hidden": 32}, 8, 0,
                     task="multiclass", output_dim=7)
    check("conf_uniform builds for multiclass", mc(torch.randn(4, 8))["pred"].shape == (4, 7))

    # 7. regression scores are rescaled to original units -- the notebook 11 bug
    check("_score rescales regression by y_std",
          abs(_score({"test_rmse": 0.4}, "regression", 1.157273) - 0.4629092) < 1e-6)
    check("_score leaves accuracy alone", _score({"test_acc": 0.9}, "multiclass", 1.0) == 0.9)

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
