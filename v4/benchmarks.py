"""
benchmarks.py  —  SRP v4

Published-split benchmark datasets, and the published results on them.

Source: Gorishniy, Rubachev & Babenko (2022), "On Embeddings for Numerical Features in Tabular
Deep Learning", NeurIPS 2022 (arXiv:2203.05556). The authors released their exact train / val /
test splits, so a model trained here can be compared with their GBDT and deep-learning results on
the same rows, with the same metric.

    data = load_benchmark("HI")          # HI = Higgs Small, FB = Facebook Comments, SA = Santander
    data.X_tr, data.y_tr, ...            # preprocessed exactly as in the authors' code
    published("HI", "XGBoost")           # -> (mean, std) single-model test score from the paper

Preprocessing follows the authors' code (tabular-dl-num-embeddings, lib/data.py):
  * numerical: QuantileTransformer(output_distribution="normal",
               n_quantiles=max(min(n_train // 30, 1000), 10), subsample=1e9), fitted on the
               training split with small noise added (1e-3 / max(std, 1e-3)), which keeps tied
               discrete values from collapsing onto the same quantile;
  * missing numerical values: training-split mean;
  * categorical: one-hot, unknown categories ignored;
  * regression target: standardised with the training mean / std. Predictions are mapped back,
    so every reported RMSE is in the ORIGINAL units, like the paper's.

Data: run `python benchmarks.py --extract` once after downloading the authors' archive
(https://www.dropbox.com/s/r0ef3ij3wl049gl/data.tar?dl=1) to v4/data/raw/num_embeddings_data.tar.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

__all__ = ["BENCHMARKS", "BenchmarkData", "load_benchmark", "published", "published_table"]

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "data" / "benchmarks"
ARCHIVE = ROOT / "data" / "raw" / "num_embeddings_data.tar"
PUBLISHED_PATH = BENCH_DIR / "published_gorishniy2022.json"

# Paper Table 1 sizes, used to verify the extracted data is the paper's split.
BENCHMARKS = {
    "HI": dict(folder="higgs-small", label="Higgs Small", sizes=(62_751, 15_688, 19_610)),
    "FB": dict(folder="fb-comments", label="Facebook Comments", sizes=(157_638, 19_722, 19_720)),
    "SA": dict(folder="santander", label="Santander", sizes=(128_000, 32_000, 40_000)),
}
_TASK = {"binclass": "binary", "multiclass": "multiclass", "regression": "regression"}
PARTS = ("train", "val", "test")


@dataclass
class BenchmarkData:
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
    y_mean: float = 0.0          # regression only: standardisation constants
    y_std: float = 1.0
    info: Dict = field(default_factory=dict)

    @property
    def output_dim(self) -> int:
        return self.n_classes if self.task == "multiclass" else 1

    @property
    def n_features(self) -> int:
        return self.X_tr.shape[1]

    def to_original(self, y_std_units: np.ndarray) -> np.ndarray:
        """Map standardised regression values back to the target's original units."""
        return np.asarray(y_std_units) * self.y_std + self.y_mean

    def y_original(self, part: str) -> np.ndarray:
        y = {"train": self.y_tr, "val": self.y_va, "test": self.y_te}[part]
        return self.to_original(y) if self.task == "regression" else y

    def arrays(self) -> Dict[str, np.ndarray]:
        ydt = np.int64 if self.task == "multiclass" else np.float32
        return {"Xtr": self.X_tr.astype(np.float32), "ytr": self.y_tr.astype(ydt),
                "Xva": self.X_va.astype(np.float32), "yva": self.y_va.astype(ydt),
                "Xte": self.X_te.astype(np.float32), "yte": self.y_te.astype(ydt)}

    def summary(self) -> str:
        s = (f"{self.key} {self.label:18s} {self.task:10s} features={self.n_features:3d} "
             f"train={len(self.y_tr):,} val={len(self.y_va):,} test={len(self.y_te):,}")
        return s + (f"  classes={self.n_classes}" if self.task != "regression" else "")


def _load_part(folder: Path, prefix: str) -> Optional[Dict[str, np.ndarray]]:
    if not (folder / f"{prefix}_train.npy").exists():
        return None
    return {p: np.load(folder / f"{prefix}_{p}.npy", allow_pickle=True) for p in PARTS}


def load_benchmark(key: str, seed: int = 0) -> BenchmarkData:
    """Load one benchmark with the authors' split and preprocessing. `seed` only affects the
    fitting noise of the quantile transform (the authors used the run seed; we fix it to 0 so
    every model sees identical inputs)."""
    meta = BENCHMARKS[key]
    folder = BENCH_DIR / meta["folder"]
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing — run `python benchmarks.py --extract` first")
    info = json.loads((folder / "info.json").read_text())
    task = _TASK[info["task_type"]]

    Xn, Xc, y = _load_part(folder, "X_num"), _load_part(folder, "X_cat"), _load_part(folder, "y")
    sizes = tuple(len(y[p]) for p in PARTS)
    if sizes != meta["sizes"]:
        raise ValueError(f"{key}: split sizes {sizes} differ from the paper's {meta['sizes']}")

    blocks = {p: [] for p in PARTS}
    if Xn is not None:
        Xn = {p: Xn[p].astype(np.float64) for p in PARTS}
        mean = np.nanmean(Xn["train"], axis=0)
        for p in PARTS:
            nan = np.isnan(Xn[p])
            if nan.any():
                Xn[p] = np.where(nan, mean, Xn[p])
        X_fit = Xn["train"]
        noise = 1e-3
        stds = np.std(X_fit, axis=0, keepdims=True)
        X_fit = X_fit + (noise / np.maximum(stds, noise)) * \
            np.random.default_rng(seed).standard_normal(X_fit.shape)
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=max(min(len(X_fit) // 30, 1000), 10),
                                 subsample=int(1e9), random_state=seed).fit(X_fit)
        for p in PARTS:
            blocks[p].append(qt.transform(Xn[p]))
    if Xc is not None:
        Xc = {p: np.asarray(Xc[p]).astype(str) for p in PARTS}     # NaN becomes its own category
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            dtype=np.float32).fit(Xc["train"])
        for p in PARTS:
            blocks[p].append(enc.transform(Xc[p]))
    X = {p: np.hstack(blocks[p]).astype(np.float32) for p in PARTS}

    y_mean, y_std, n_classes = 0.0, 1.0, 0
    if task == "regression":
        y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
        y = {p: ((y[p] - y_mean) / y_std).astype(np.float32) for p in PARTS}
    else:
        y = {p: y[p].astype(np.int64) for p in PARTS}
        n_classes = int(max(y[p].max() for p in PARTS)) + 1

    return BenchmarkData(key=key, label=meta["label"], task=task,
                         X_tr=X["train"], y_tr=y["train"], X_va=X["val"], y_va=y["val"],
                         X_te=X["test"], y_te=y["test"], n_classes=n_classes,
                         y_mean=y_mean, y_std=y_std,
                         info={**info, "n_num": 0 if Xn is None else Xn["train"].shape[1],
                               "n_cat_onehot": 0 if Xc is None else blocks["train"][-1].shape[1]})


# ---------------------------------------------------------------------------
# Published results
# ---------------------------------------------------------------------------
def _published() -> Dict:
    return json.loads(PUBLISHED_PATH.read_text())


def published(key: str, model: str, kind: str = "single") -> Tuple[float, float]:
    """(mean, std) test score from the paper. kind = "single" (15 seeds) or "ensemble"."""
    return tuple(_published()[kind][model][key])


def published_table(keys=("HI", "FB", "SA"), kind: str = "single"):
    import pandas as pd
    pub = _published()[kind]
    return pd.DataFrame({k: {m: v[k][0] for m, v in pub.items() if k in v} for k in keys})


def higher_is_better(key: str) -> bool:
    return load_task_type(key) != "regression"


def load_task_type(key: str) -> str:
    info = json.loads((BENCH_DIR / BENCHMARKS[key]["folder"] / "info.json").read_text())
    return _TASK[info["task_type"]]


# ---------------------------------------------------------------------------
# One-time extraction from the authors' archive
# ---------------------------------------------------------------------------
def extract(archive: Path = ARCHIVE):
    """Pull only our three datasets out of the 3.3 GB archive into data/benchmarks/<folder>/."""
    import tarfile
    wanted = {f"data/{m['folder']}/" for m in BENCHMARKS.values()}
    with tarfile.open(archive) as tar:
        # Only the prepared split files; the archive also carries each dataset's raw source
        # files (Santander's Kaggle zip/csv, Facebook's original CSVs), which we don't need.
        members = [m for m in tar.getmembers()
                   if m.isfile() and any(m.name.startswith(w) for w in wanted)
                   and m.name.endswith((".npy", ".json"))]
        for m in members:
            m.name = m.name[len("data/"):]
        tar.extractall(BENCH_DIR, members=members)
    for key in BENCHMARKS:
        print(load_benchmark(key).summary())


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    else:
        for key in BENCHMARKS:
            print(load_benchmark(key).summary())
