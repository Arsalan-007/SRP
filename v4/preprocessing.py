"""
preprocessing.py  —  SRP v4

The published-split preprocessing recipe, in one place.

`benchmarks.py` (Gorishniy et al. 2022) and `benchmarks_rtdl.py` (Gorishniy et al. 2021)
load different archives with different folder conventions, but they preprocess the arrays
IDENTICALLY — both reproduce the papers' own pipeline so our numbers sit next to theirs:

  * numerical: QuantileTransformer(output_distribution="normal",
    n_quantiles=max(min(n_train // 30, 1000), 10), subsample=1e9), fitted on the TRAINING
    split with a small noise term (1e-3 / max(std, 1e-3)) added first, which keeps tied
    discrete values from collapsing onto the same quantile;
  * missing numerical values: training-split mean;
  * categorical: one-hot, unknown categories ignored, NaN kept as its own category;
  * regression target: standardised with the training mean / std, so predictions can be
    mapped back and every reported RMSE is in the ORIGINAL units, like the papers'.

Until this module existed that recipe was written out twice, once per loader, and the two
copies had to be kept in step by hand.

NOT USED BY `datasets.py`, deliberately. That module loads our own diagnostic splits and
uses a DIFFERENT recipe: it separates binary from continuous columns and only standardises
the binary ones, because a quantile transform maps a 0/1 column onto two extreme values
(about ±5) — which would wreck Covertype's 44 one-hot soil indicators and, more to the
point, the rotation diagnostic that depends on every column being on a comparable scale.
Two different jobs, two different recipes; merging them would be wrong.

Self-test:  python preprocessing.py
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

__all__ = ["PARTS", "preprocess_splits", "standardise_target"]

PARTS = ("train", "val", "test")


def preprocess_splits(num: Optional[Dict[str, np.ndarray]],
                      cat: Optional[Dict[str, np.ndarray]],
                      seed: int = 0) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Numerical + categorical blocks -> one float32 matrix per split.

    `num` / `cat` map each split name in PARTS to its raw array, or are None when the
    dataset has no columns of that kind. Everything is fitted on "train" only.

    Returns (X per split, n_numerical_columns, n_onehot_columns).
    """
    if num is None and cat is None:
        raise ValueError("a dataset needs numerical or categorical columns, got neither")

    blocks: Dict[str, list] = {p: [] for p in PARTS}
    n_num = n_onehot = 0

    if num is not None:
        num = {p: np.asarray(num[p]).astype(np.float64) for p in PARTS}
        mean = np.nanmean(num["train"], axis=0)
        for p in PARTS:
            nan = np.isnan(num[p])
            if nan.any():
                num[p] = np.where(nan, mean, num[p])
        X_fit = num["train"]
        noise = 1e-3
        stds = np.std(X_fit, axis=0, keepdims=True)
        X_fit = X_fit + (noise / np.maximum(stds, noise)) * \
            np.random.default_rng(seed).standard_normal(X_fit.shape)
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=max(min(len(X_fit) // 30, 1000), 10),
                                 subsample=int(1e9), random_state=seed).fit(X_fit)
        for p in PARTS:
            blocks[p].append(qt.transform(num[p]))
        n_num = num["train"].shape[1]

    if cat is not None:
        cat = {p: np.asarray(cat[p]).astype(str) for p in PARTS}   # NaN becomes its own category
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            dtype=np.float32).fit(cat["train"])
        for p in PARTS:
            blocks[p].append(enc.transform(cat[p]))
        n_onehot = blocks["train"][-1].shape[1]

    X = {p: np.hstack(blocks[p]).astype(np.float32) for p in PARTS}
    return X, n_num, n_onehot


def standardise_target(y: Dict[str, np.ndarray], task: str
                       ) -> Tuple[Dict[str, np.ndarray], float, float, int]:
    """Regression targets -> z-scored with the TRAIN mean/std; classification -> int64.

    Returns (y per split, y_mean, y_std, n_classes). For classification y_mean/y_std are
    the identity (0.0, 1.0) so callers can rescale unconditionally.
    """
    if task == "regression":
        y_mean, y_std = float(y["train"].mean()), float(y["train"].std())
        return {p: ((y[p] - y_mean) / y_std).astype(np.float32) for p in PARTS}, y_mean, y_std, 0
    y = {p: np.asarray(y[p]).astype(np.int64) for p in PARTS}
    return y, 0.0, 1.0, int(max(y[p].max() for p in PARTS)) + 1


# ---------------------------------------------------------------------------
# Self-test:  python preprocessing.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = {"train": rng.normal(size=(600, 4)), "val": rng.normal(size=(200, 4)),
         "test": rng.normal(size=(200, 4))}
    n["train"][5, 2] = np.nan                       # a hole to impute
    c = {p: rng.choice(["a", "b", "c"], size=(len(n[p]), 2)) for p in PARTS}

    X, n_num, n_oh = preprocess_splits(n, c, seed=0)
    ok1 = all(X[p].dtype == np.float32 for p in PARTS) and not np.isnan(X["train"]).any()
    print(f"  [{'PASS' if ok1 else 'FAIL'}] output is float32 with NaNs imputed from the train mean")
    ok2 = n_num == 4 and n_oh == 6 and X["train"].shape[1] == 10
    print(f"  [{'PASS' if ok2 else 'FAIL'}] numerical({n_num}) + one-hot({n_oh}) == {X['train'].shape[1]} columns")

    # Fitted on train only: refitting with a different val/test must not change train.
    n2 = dict(n); n2["test"] = rng.normal(size=(200, 4)) * 50
    X2, _, _ = preprocess_splits(n2, c, seed=0)
    ok3 = np.array_equal(X["train"], X2["train"])
    print(f"  [{'PASS' if ok3 else 'FAIL'}] transform is fitted on TRAIN only (a wild test split "
          f"does not move the train rows)")

    ok4 = np.array_equal(X["train"], preprocess_splits(n, c, seed=0)[0]["train"])
    print(f"  [{'PASS' if ok4 else 'FAIL'}] same seed -> bit-identical output (the noise term is seeded)")
    ok5 = not np.array_equal(X["train"], preprocess_splits(n, c, seed=1)[0]["train"])
    print(f"  [{'PASS' if ok5 else 'FAIL'}] a different seed moves the output (noise is really applied)")

    yr = {"train": rng.normal(size=600) * 3 + 10, "val": rng.normal(size=200),
          "test": rng.normal(size=200)}
    ys, ym, ysd, ncl = standardise_target(yr, "regression")
    ok6 = abs(ys["train"].mean()) < 1e-5 and abs(ys["train"].std() - 1) < 1e-4 and ncl == 0
    print(f"  [{'PASS' if ok6 else 'FAIL'}] regression target standardised on train "
          f"(mean {ys['train'].mean():+.1e}, std {ys['train'].std():.4f})")
    back = ys["train"] * ysd + ym
    ok7 = np.allclose(back, yr["train"], atol=1e-4)
    print(f"  [{'PASS' if ok7 else 'FAIL'}] y_mean/y_std map predictions back to ORIGINAL units")

    yc = {p: rng.integers(0, 3, size=len(n[p])) for p in PARTS}
    yi, ym2, ysd2, ncl2 = standardise_target(yc, "multiclass")
    ok8 = yi["train"].dtype == np.int64 and ncl2 == 3 and (ym2, ysd2) == (0.0, 1.0)
    print(f"  [{'PASS' if ok8 else 'FAIL'}] classification target -> int64, {ncl2} classes, identity rescale")

    ok9 = preprocess_splits(n, None, seed=0)[0]["train"].shape[1] == 4 and \
          preprocess_splits(None, c, seed=0)[0]["train"].shape[1] == 6
    print(f"  [{'PASS' if ok9 else 'FAIL'}] numerical-only and categorical-only datasets both work")
    try:
        preprocess_splits(None, None); ok10 = False
    except ValueError:
        ok10 = True
    print(f"  [{'PASS' if ok10 else 'FAIL'}] a dataset with no columns at all raises instead of "
          f"silently returning an empty matrix")
