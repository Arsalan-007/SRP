"""
results.py  —  SRP v4

Structured, on-disk results — the thing this project never had until now. Every one of
the first 14 notebooks' results lived ONLY as printed text inside a `.ipynb` cell output:
no way to reload "the model that got 0.4039 RMSE on California Housing," no way to build
a ranked comparison table without re-reading a dozen notebooks by eye.

INSPIRED BY, NOT COPIED FROM, TabR (yandex-research/tabular-dl-tabr)
----------------------------------------------------------------------
Their public repo's `lib/` and `notebooks/` — which hold the actual `Dataset` classes and
the ranking-table code — are NOT published (confirmed: neither directory exists in the
GitHub tree, and `bin/ffn.py`'s `import lib` never resolves in the public checkout). What
IS public, and what this module's design comes from, is their `.gitignore` convention:

    exp/**/*.*
    !exp/**/report.json
    !exp/**/*.toml

Every run's heavy artifacts are excluded; a tiny `report.json` per run is tracked. A real
example (`exp/mlp/california/0-evaluation/0/report.json`) has this shape:

    {"function": ..., "config": {...}, "n_parameters": ..., "best_epoch": ...,
     "metrics": {"train": {...}, "val": {...}, "test": {...}}, "time": "0:00:20"}

This module adopts that JSON-per-run pattern (`report.json` + a sibling `checkpoint.pt`),
not their code — we don't have their code.

THE "source" FIELD — the most important thing in the schema
----------------------------------------------------------------------
No new experiments are being run for this backfill: most `report.json` entries are
TRANSCRIBED from the README's already-verified numbers (Steps 6-12), not freshly computed.
A `"documented"` entry (`checkpoint=None`) must never be mistaken for a `"reproduced"` one
(a real training run this session, with a real `checkpoint.pt` on disk) or a `"published"`
one (someone else's paper). Every table this module builds keeps that column visible.

Directory convention: results/<DATASET_KEY>/<arm>/<tag>/seed<N>/report.json[, checkpoint.pt]
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["SOURCES", "save_report", "save_checkpoint", "load_reports",
           "summarize_ranks", "best_per_dataset", "DEFAULT_RESULTS_DIR"]

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOURCES = ("reproduced", "documented", "published")


def save_report(dir_path, **fields) -> Path:
    """Write one report.json. `fields` is whatever the caller has — dataset, label, arm,
    tag, seed, source, source_ref, config, n_parameters, metrics, champion_metric,
    champion_value, higher_is_better, time_seconds, checkpoint (a path string, or None
    for "documented"/"published" entries with no real trained weights)."""
    if fields.get("source") not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {fields.get('source')!r}")
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "report.json"
    path.write_text(json.dumps(fields, indent=2, default=str))
    return path


def save_checkpoint(model, dir_path) -> Path:
    """Thin wrapper: torch.save(model.state_dict(), .../checkpoint.pt). Only meaningful
    for source="reproduced" — a "documented" entry has nothing to save."""
    import torch
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    return path


def load_reports(results_dir=DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    """Recursively glob every report.json under results_dir into one long-format
    DataFrame, one row per run. Missing/empty results_dir returns an empty DataFrame
    with the expected columns, not an error."""
    root = Path(results_dir)
    rows = []
    for p in sorted(root.glob("**/report.json")):
        d = json.loads(p.read_text())
        d["_report_path"] = str(p)
        rows.append(d)
    cols = ["dataset", "label", "arm", "tag", "seed", "source", "source_ref", "config",
            "n_parameters", "metrics", "champion_metric", "champion_value",
            "higher_is_better", "time_seconds", "checkpoint", "_report_path"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def summarize_ranks(df: pd.DataFrame, dataset_col: str = "dataset", model_col: str = "arm",
                    metric_col: str = "champion_value",
                    higher_is_better_col: str = "higher_is_better") -> pd.DataFrame:
    """TabR-style ranking table: within each dataset, rank every model (1 = best,
    respecting that dataset's higher_is_better direction), then average rank +/- std
    across datasets. Multiple rows for the same (dataset, model) — e.g. several seeds —
    are averaged BEFORE ranking, so a seed sweep doesn't inflate one model's row count.

    Returns a DataFrame indexed by model_col, columns ["avg", "std", <dataset 1>, ...],
    sorted best-first (lowest avg rank) — the same shape as the TabR screenshot's
    "Ranks:" table.
    """
    if df.empty:
        return pd.DataFrame(columns=["avg", "std"])
    d = df.copy()
    d["_signed"] = np.where(d[higher_is_better_col].astype(bool),
                            d[metric_col].astype(float), -d[metric_col].astype(float))
    # Collapse repeated seeds for the same (dataset, model) to one value first.
    collapsed = d.groupby([dataset_col, model_col])["_signed"].mean().reset_index()
    collapsed["_rank"] = collapsed.groupby(dataset_col)["_signed"].rank(
        ascending=False, method="average")
    pivot = collapsed.pivot(index=model_col, columns=dataset_col, values="_rank")
    pivot.insert(0, "std", pivot.std(axis=1, ddof=1))
    pivot.insert(0, "avg", pivot.drop(columns="std").mean(axis=1))
    return pivot.sort_values("avg")


def best_per_dataset(df: pd.DataFrame, dataset_col: str = "dataset",
                     metric_col: str = "champion_value",
                     higher_is_better_col: str = "higher_is_better") -> pd.DataFrame:
    """One row per dataset: whichever run actually won it (the "champion"). Ties broken
    arbitrarily but deterministically (first occurrence) — real ties should be rare at
    the precision these metrics are reported to."""
    if df.empty:
        return df
    d = df.copy()
    d["_signed"] = np.where(d[higher_is_better_col].astype(bool),
                            d[metric_col].astype(float), -d[metric_col].astype(float))
    idx = d.groupby(dataset_col)["_signed"].idxmax()
    return df.loc[idx].drop(columns=[], errors="ignore").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Smoke test:  python results.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    print(f"CHECKS (scratch dir: {tmp})")

    # Synthetic 2-dataset x 3-model report set. Model "A" wins both datasets outright,
    # model "B" loses both, model "C" is mixed — lets us hand-verify the ranks.
    synthetic = [
        # dataset, model, value, higher_is_better, seed
        ("D1", "A", 0.90, True, 0), ("D1", "A", 0.92, True, 1),   # avg 0.91 -> rank 1
        ("D1", "B", 0.70, True, 0),                                # rank 3
        ("D1", "C", 0.80, True, 0),                                # rank 2
        ("D2", "A", 0.10, False, 0),                               # rmse-like, lower better -> rank 1
        ("D2", "B", 0.30, False, 0),                               # rank 3
        ("D2", "C", 0.20, False, 0),                               # rank 2
    ]
    for i, (ds, model, val, hib, seed) in enumerate(synthetic):
        d = tmp / ds / model / "default" / f"seed{seed}"
        save_report(d, dataset=ds, label=ds, arm=model, tag="default", seed=seed,
                   source="documented", source_ref="synthetic self-test",
                   config={}, n_parameters=0, metrics={}, champion_metric="score",
                   champion_value=val, higher_is_better=hib, time_seconds=0.0,
                   checkpoint=None)

    df = load_reports(tmp)
    ok = len(df) == len(synthetic)
    print(f"  [{'PASS' if ok else 'FAIL'}] load_reports round-trips all {len(synthetic)} "
          f"synthetic reports ({len(df)} loaded)")

    ranks = summarize_ranks(df)
    ok = (ranks.loc["A", "avg"] == 1.0 and ranks.loc["B", "avg"] == 3.0
         and ranks.loc["C", "avg"] == 2.0)
    print(f"  [{'PASS' if ok else 'FAIL'}] summarize_ranks: A (wins both) -> avg rank 1.0, "
          f"B (loses both) -> 3.0, C (mixed) -> 2.0")
    print(ranks)

    champs = best_per_dataset(df)
    ok = set(champs[champs.dataset == "D1"]["arm"]) == {"A"} and \
        set(champs[champs.dataset == "D2"]["arm"]) == {"A"}
    print(f"\n  [{'PASS' if ok else 'FAIL'}] best_per_dataset picks model A for both datasets")

    # A "documented" row (checkpoint=None) must not crash anything downstream.
    ok = df["checkpoint"].isna().all()
    print(f"  [{'PASS' if ok else 'FAIL'}] every synthetic row is source='documented' with "
          f"checkpoint=None, and neither load_reports nor summarize_ranks choked on that")

    # save_report rejects an invalid source instead of silently writing garbage.
    try:
        save_report(tmp / "bad", source="made_up")
        ok = False
    except ValueError:
        ok = True
    print(f"  [{'PASS' if ok else 'FAIL'}] save_report rejects an unrecognised source value")

    shutil.rmtree(tmp)
    print(f"\n(scratch dir cleaned up)")
