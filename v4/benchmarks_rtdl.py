"""
benchmarks_rtdl.py  —  SRP v4

Published-split datasets from Gorishniy, Rubachev, Khrulkov & Babenko (2021), "Revisiting
Deep Learning Models for Tabular Data", NeurIPS 2021 (arXiv:2106.11959) — the FT-Transformer
paper. Same design as benchmarks.py (which covers a different paper's 3 datasets); kept as a
separate module so it can't disturb that already-validated pipeline.

    data = load_benchmark("CA")          # CA=California Housing, AD=Adult, JA=Jannis,
    data.X_tr, data.y_tr, ...            # CO=Covertype, YE=YearPredictionMSD
    published("CA", "XGBoost", kind="gbdt")   # -> mean score from the paper's GBDT table

Preprocessing mirrors benchmarks.py: numerical features get a QuantileTransformer fitted on
train only (with the authors' small fitting-noise trick to break ties), categorical features
(Adult only) get one-hot encoding fitted on train only, and regression targets are
standardised — RMSE is always reported back in the target's ORIGINAL units.

Data: run `python benchmarks_rtdl.py --extract` once, after downloading the authors' archive
(https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1, ~4.1 GB) to
v4/data/raw/revisiting_models_data.tar.gz, to populate v4/data/benchmarks_rtdl/<folder>/.
That extraction has already been done once for this project; data/benchmarks_rtdl/ is what
ships to the GPU server (see sync_to_server.sh), NOT data/raw/.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

__all__ = ["BENCHMARKS", "RtdlBenchmarkData", "load_benchmark", "published",
           "published_table", "higher_is_better", "load_task_type"]

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "data" / "benchmarks_rtdl"
ARCHIVE = ROOT / "data" / "raw" / "revisiting_models_data.tar.gz"
PUBLISHED_PATH = BENCH_DIR / "published_gorishniy2021.json"

# Paper's published split sizes, used to verify the extracted data is the paper's split
# (same defensive check as benchmarks.py — a silent split mismatch would poison every
# downstream comparison).
BENCHMARKS = {
    "CA": dict(folder="california_housing", label="California Housing",
              sizes=(13_209, 3_303, 4_128)),
    "AD": dict(folder="adult", label="Adult", sizes=(26_048, 6_513, 16_281)),
    "JA": dict(folder="jannis", label="Jannis", sizes=(53_588, 13_398, 16_747)),
    "CO": dict(folder="covtype", label="Covertype", sizes=(371_847, 92_962, 116_203)),
    "YE": dict(folder="year", label="YearPredictionMSD", sizes=(370_972, 92_743, 51_630)),
}
_TASK = {"binclass": "binary", "multiclass": "multiclass", "regression": "regression"}
PARTS = ("train", "val", "test")


@dataclass
class RtdlBenchmarkData:
    key: str
    label: str
    task: str
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    n_classes: int = 0
    y_mean: float = 0.0
    y_std: float = 1.0
    info: Dict = field(default_factory=dict)

    @property
    def output_dim(self) -> int:
        return self.n_classes if self.task == "multiclass" else 1

    @property
    def n_features(self) -> int:
        return self.X_tr.shape[1]

    def to_original(self, y_std_units: np.ndarray) -> np.ndarray:
        return np.asarray(y_std_units) * self.y_std + self.y_mean

    def arrays(self) -> Dict[str, np.ndarray]:
        ydt = np.int64 if self.task == "multiclass" else np.float32
        return {"Xtr": self.X_tr.astype(np.float32), "ytr": self.y_tr.astype(ydt),
                "Xva": self.X_va.astype(np.float32), "yva": self.y_va.astype(ydt),
                "Xte": self.X_te.astype(np.float32), "yte": self.y_te.astype(ydt)}

    def summary(self) -> str:
        s = (f"{self.key} {self.label:18s} {self.task:10s} features={self.n_features:3d} "
             f"train={len(self.y_tr):,} val={len(self.y_va):,} test={len(self.y_te):,}")
        return s + (f"  classes={self.n_classes}" if self.task != "regression" else "")


def _load(folder: Path, prefix: str) -> Optional[Dict[str, np.ndarray]]:
    if not (folder / f"{prefix}_train.npy").exists():
        return None
    return {p: np.load(folder / f"{prefix}_{p}.npy", allow_pickle=True) for p in PARTS}


def load_benchmark(key: str, seed: int = 0) -> RtdlBenchmarkData:
    """Load one benchmark with the authors' split and preprocessing. `seed` only affects the
    quantile-transform fitting noise (fixed to 0 so every model sees identical inputs)."""
    meta = BENCHMARKS[key]
    folder = BENCH_DIR / meta["folder"]
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing — run `python benchmarks_rtdl.py --extract` first")
    info = json.loads((folder / "info.json").read_text())
    task = _TASK[info["task_type"]]

    Nn, Cc, y = _load(folder, "N"), _load(folder, "C"), _load(folder, "y")
    sizes = tuple(len(y[p]) for p in PARTS)
    if sizes != meta["sizes"]:
        raise ValueError(f"{key}: split sizes {sizes} differ from the paper's {meta['sizes']}")

    blocks = {p: [] for p in PARTS}
    if Nn is not None:
        Nn = {p: Nn[p].astype(np.float64) for p in PARTS}
        mean = np.nanmean(Nn["train"], axis=0)
        for p in PARTS:
            nan = np.isnan(Nn[p])
            if nan.any():
                Nn[p] = np.where(nan, mean, Nn[p])
        X_fit = Nn["train"]
        noise = 1e-3
        stds = np.std(X_fit, axis=0, keepdims=True)
        X_fit = X_fit + (noise / np.maximum(stds, noise)) * \
            np.random.default_rng(seed).standard_normal(X_fit.shape)
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=max(min(len(X_fit) // 30, 1000), 10),
                                 subsample=int(1e9), random_state=seed).fit(X_fit)
        for p in PARTS:
            blocks[p].append(qt.transform(Nn[p]))
    if Cc is not None:
        Cc = {p: np.asarray(Cc[p]).astype(str) for p in PARTS}   # includes the string "nan"
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            dtype=np.float32).fit(Cc["train"])
        for p in PARTS:
            blocks[p].append(enc.transform(Cc[p]))
    X = {p: np.hstack(blocks[p]).astype(np.float32) for p in PARTS}

    y_mean, y_std, n_classes = 0.0, 1.0, 0
    if task == "regression":
        y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
        y = {p: ((y[p] - y_mean) / y_std).astype(np.float32) for p in PARTS}
    else:
        y = {p: y[p].astype(np.int64) for p in PARTS}
        n_classes = int(max(y[p].max() for p in PARTS)) + 1

    return RtdlBenchmarkData(key=key, label=meta["label"], task=task,
                             X_tr=X["train"], y_tr=y["train"], X_va=X["val"], y_va=y["val"],
                             X_te=X["test"], y_te=y["test"], n_classes=n_classes,
                             y_mean=y_mean, y_std=y_std,
                             info={**info, "n_num": 0 if Nn is None else Nn["train"].shape[1],
                                   "n_cat_onehot": 0 if Cc is None else blocks["train"][-1].shape[1]})


# ---------------------------------------------------------------------------
# Published results
# ---------------------------------------------------------------------------
def _published() -> Dict:
    return json.loads(PUBLISHED_PATH.read_text())


def published(key: str, model: str, kind: str = "single") -> float:
    """Score from the paper. kind = 'single' (one tuned model), 'ensemble' (of those single
    models), or 'gbdt' (separately tuned XGBoost/CatBoost). No std reported for this paper's
    tables (unlike benchmarks.py's paper), so this returns a single float, not (mean, std)."""
    return _published()[kind][model][key]


def published_table(keys=("CA", "AD", "JA", "CO", "YE"), kind: str = "single"):
    import pandas as pd
    pub = _published()[kind]
    return pd.DataFrame({k: {m: v[k] for m, v in pub.items() if k in v} for k in keys})


def higher_is_better(key: str) -> bool:
    return load_task_type(key) != "regression"


def load_task_type(key: str) -> str:
    info = json.loads((BENCH_DIR / BENCHMARKS[key]["folder"] / "info.json").read_text())
    return _TASK[info["task_type"]]


# ---------------------------------------------------------------------------
# One-time extraction from the authors' archive
# ---------------------------------------------------------------------------
def extract(archive: Path = ARCHIVE):
    """Pull only our 5 datasets out of the ~4.1 GB archive into data/benchmarks_rtdl/<folder>/."""
    import shutil
    import tarfile
    import tempfile
    wanted = {m["folder"] for m in BENCHMARKS.values()}
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers()
                  if any(m.name.startswith(f"data/{f}/") for f in wanted)]
        tar.extractall(tmp, members=members)
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        for f in wanted:
            src, dst = Path(tmp) / "data" / f, BENCH_DIR / f
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            (dst / "READY").touch()
    print(f"extracted {len(wanted)} datasets to {BENCH_DIR}")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    for k in BENCHMARKS:
        t = load_benchmark(k)
        sizes = (len(t.y_tr), len(t.y_va), len(t.y_te))
        ok = sizes == tuple(BENCHMARKS[k]["sizes"])
        print(f"{t.summary()}  | split sizes match paper: {ok}")
        for kind in ("single", "ensemble", "gbdt"):
            try:
                v = published(k, "XGBoost" if kind == "gbdt" else "FT-Transformer", kind=kind)
                print(f"   published {kind:9s} {'XGBoost' if kind=='gbdt' else 'FT-Transformer':16s} {v}")
            except KeyError:
                pass
