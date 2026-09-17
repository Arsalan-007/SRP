"""
datasets.py  —  SRP v4

Every benchmark dataset behind one interface: download once, cache, draw a fixed-size
random sample, split train / val / test, and preprocess with statistics fitted on the
training split only.

    task = make_task("covertype", n_train=10_000, seed=0)
    task.X_tr, task.y_tr, task.X_va, task.y_va, task.X_te, task.y_te
    task.task          -> "regression" | "binary" | "multiclass"
    task.output_dim    -> 1 for regression/binary, n_classes for multiclass

Why fixed-size samples instead of the full data
-----------------------------------------------
Grinsztajn et al. (2022) ran their main benchmark with training sets capped at 10,000
rows — the "medium-sized" regime in which they showed trees beating neural networks.
Matching that regime is what makes our diagnostics comparable to their findings. It is
also the only feasible option on CPU: one causal-attention run on 12k rows takes ~8 min,
so a single epoch over HIGGS's 11M rows would take ~40 min. `n_train` is a parameter;
raise it on a GPU.

California Housing is the exception: it always uses the exact 60/20/20 split of
notebooks 01-03 so its numbers stay comparable with everything before.
"""

import gzip
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

__all__ = ["TabularTask", "make_task", "load_raw", "DATASETS"]

DATA_DIR = Path(__file__).resolve().parent / "data"

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
POKER_URL = "https://archive.ics.uci.edu/static/public/158/poker+hand.zip"
# HIGGS is 2.6 GB compressed and the server does not support partial downloads, so we
# stream and keep only the first HIGGS_ROWS rows (the file's rows are in random order).
HIGGS_ROWS = 500_000

DATASETS = {
    "california": dict(task="regression", full_rows=20_640, source="sklearn"),
    "covertype":  dict(task="multiclass", full_rows=581_012, source="sklearn (UCI covtype)"),
    "poker":      dict(task="multiclass", full_rows=1_025_010, source="UCI Poker Hand"),
    "higgs":      dict(task="binary", full_rows=11_000_000,
                       source=f"UCI HIGGS (first {HIGGS_ROWS:,} rows streamed)"),
}

_HIGGS_NAMES = (["lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_mag", "missing_energy_phi"]
                + [f"jet{j}_{q}" for j in range(1, 5) for q in ("pt", "eta", "phi", "btag")]
                + ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"])


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class TabularTask:
    name: str
    task: str
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    feature_names: List[str]
    n_classes: int = 0
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X_tr.shape[1]

    @property
    def output_dim(self) -> int:
        return self.n_classes if self.task == "multiclass" else 1

    @property
    def majority_acc(self) -> Optional[float]:
        """Test accuracy of always predicting the most frequent TRAINING class."""
        if self.task == "regression":
            return None
        mode = np.bincount(self.y_tr.astype(int)).argmax()
        return float((self.y_te.astype(int) == mode).mean())

    def arrays(self) -> Dict[str, np.ndarray]:
        """Split arrays with training-ready dtypes (int64 labels for multiclass)."""
        ydt = np.int64 if self.task == "multiclass" else np.float32
        return {"Xtr": self.X_tr.astype(np.float32), "ytr": self.y_tr.astype(ydt),
                "Xva": self.X_va.astype(np.float32), "yva": self.y_va.astype(ydt),
                "Xte": self.X_te.astype(np.float32), "yte": self.y_te.astype(ydt)}

    def with_(self, **changes) -> "TabularTask":
        return replace(self, **changes)

    def summary(self) -> str:
        s = (f"{self.name:11s} {self.task:10s} features={self.n_features:3d} "
             f"train={len(self.y_tr):,} val={len(self.y_va):,} test={len(self.y_te):,}")
        if self.task != "regression":
            s += f"  classes={self.n_classes}  majority-class acc={self.majority_acc:.3f}"
        return s


# ---------------------------------------------------------------------------
# Raw loaders (cached)
# ---------------------------------------------------------------------------
def _load_california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    return d.data.astype(np.float64), d.target.astype(np.float64), list(d.feature_names)


def _load_covertype():
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype()
    names = list(getattr(d, "feature_names", [f"f{i}" for i in range(d.data.shape[1])]))
    return d.data.astype(np.float64), (d.target - 1).astype(np.int64), names   # 1..7 -> 0..6


def _load_poker():
    cache = DATA_DIR / "poker.npz"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(POKER_URL, timeout=120) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        parts = [np.loadtxt(zf.open(f), delimiter=",", dtype=np.int64)
                 for f in ("poker-hand-training-true.data", "poker-hand-testing.data")]
        arr = np.vstack(parts)
        np.savez_compressed(cache, data=arr)
    arr = np.load(cache)["data"]
    names = [f"{kind}{i}" for i in range(1, 6) for kind in ("suit", "rank")]
    return arr[:, :10].astype(np.float64), arr[:, 10].astype(np.int64), names


def _load_higgs():
    cache = DATA_DIR / f"higgs_first{HIGGS_ROWS}.npy"
    if not cache.exists():
        import pandas as pd
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HIGGS_URL, timeout=120) as r, gzip.GzipFile(fileobj=r) as gz:
            df = pd.read_csv(gz, header=None, nrows=HIGGS_ROWS, dtype=np.float32)
        np.save(cache, df.values)
    arr = np.load(cache)
    return arr[:, 1:].astype(np.float64), arr[:, 0].astype(np.int64), list(_HIGGS_NAMES)


_LOADERS = {"california": _load_california, "covertype": _load_covertype,
            "poker": _load_poker, "higgs": _load_higgs}
_RAW_CACHE: Dict[str, Tuple] = {}


def load_raw(name: str):
    """(X, y, feature_names) for the full (or, for HIGGS, streamed) dataset. Cached."""
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_LOADERS)}")
    if name not in _RAW_CACHE:
        _RAW_CACHE[name] = _LOADERS[name]()
    return _RAW_CACHE[name]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _fit_preprocessor(X_tr: np.ndarray, seed: int):
    """Continuous columns -> QuantileTransformer(normal); binary columns -> z-score.

    A quantile transform on a 0/1 column would map it to two extreme values (about ±5),
    so binary columns (e.g. Covertype's 44 one-hot soil/wilderness indicators) are only
    standardised. Either way every column ends up on a comparable unit scale, which
    matters for the rotation diagnostic.
    """
    n_unique = np.array([len(np.unique(X_tr[:, j])) for j in range(X_tr.shape[1])])
    binary = n_unique <= 2
    cont = ~binary
    qt = None
    if cont.any():
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=min(1000, len(X_tr)), random_state=seed)
        qt.fit(X_tr[:, cont])
    mu = X_tr[:, binary].mean(0) if binary.any() else None
    sd = X_tr[:, binary].std(0) if binary.any() else None
    if sd is not None:
        sd = np.where(sd < 1e-12, 1.0, sd)

    def transform(X):
        out = np.empty_like(X, dtype=np.float64)
        if cont.any():
            out[:, cont] = qt.transform(X[:, cont])
        if binary.any():
            out[:, binary] = (X[:, binary] - mu) / sd
        return out

    return transform, {"n_continuous": int(cont.sum()), "n_binary": int(binary.sum())}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def make_task(name: str, n_train: int = 10_000, n_val: int = 5_000, n_test: int = 20_000,
              seed: int = 0) -> TabularTask:
    """Sample, split and preprocess one dataset.

    `seed` fixes WHICH rows are drawn. Keep it constant across model seeds so every model
    and every diagnostic condition sees exactly the same data.
    """
    X, y, names = load_raw(name)
    kind = DATASETS[name]["task"]

    if name == "california":
        # Identical to notebooks 01-03: 60/20/20 with random_state=42.
        X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)
        pre_seed = 42
    else:
        total = n_train + n_val + n_test
        if total > len(X):
            raise ValueError(f"{name} has {len(X):,} rows; asked for {total:,}")
        idx = np.random.RandomState(seed).choice(len(X), size=total, replace=False)
        tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
        X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
        pre_seed = seed

    info = {"full_rows": len(X), "source": DATASETS[name]["source"]}
    n_classes = 0
    if kind != "regression":
        # Re-index labels to the classes present in the TRAINING sample. A class too rare
        # to appear in train (e.g. Poker's royal flush: ~8 in 1M hands) cannot be learned
        # by any model; its few val/test rows are dropped and counted.
        classes = np.unique(y_tr)
        lut = {c: i for i, c in enumerate(classes)}
        keep_va, keep_te = np.isin(y_va, classes), np.isin(y_te, classes)
        info["dropped_unseen_class_rows"] = int((~keep_va).sum() + (~keep_te).sum())
        X_va, y_va, X_te, y_te = X_va[keep_va], y_va[keep_va], X_te[keep_te], y_te[keep_te]
        y_tr, y_va, y_te = (np.vectorize(lut.get)(a) if len(a) else a for a in (y_tr, y_va, y_te))
        n_classes = len(classes)
        info["train_class_counts"] = np.bincount(y_tr).tolist()
        if kind == "binary" and n_classes != 2:
            raise ValueError(f"{name}: expected 2 classes in train, found {n_classes}")

    transform, pinfo = _fit_preprocessor(X_tr, pre_seed)
    info.update(pinfo)
    return TabularTask(name=name, task=kind,
                       X_tr=transform(X_tr), y_tr=y_tr, X_va=transform(X_va), y_va=y_va,
                       X_te=transform(X_te), y_te=y_te,
                       feature_names=list(names), n_classes=n_classes, info=info)


if __name__ == "__main__":
    import time
    for name in DATASETS:
        t = time.time()
        task = make_task(name)
        print(f"{task.summary()}   ({time.time() - t:.1f}s)")
        print(f"            {task.info}")
