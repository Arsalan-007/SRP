"""
diagnostics.py  —  SRP v4

The three perturbation tests of Grinsztajn, Oyallon & Varoquaux (2022),
"Why do tree-based models still outperform deep learning on typical tabular data?"

Each function takes a preprocessed `TabularTask` and returns a perturbed copy. Every
model is then trained on the perturbed task with its hyperparameters unchanged, and the
CHANGE in test score versus the unperturbed task is the measurement.

    add_noise_features  robustness to uninformative features
                        -> a model that is robust barely moves
    rotate_features     does the model use the feature axes?
                        -> a rotation-invariant model (e.g. an MLP) does not move at all;
                           a model that exploits individual columns (trees) gets worse.
                           Here a DROP is the good sign.
    smooth_targets      can the model learn irregular / jumpy functions?
                        -> smooth the TRAINING labels; a model that was exploiting sharp,
                           irregular structure loses a lot, a model that could only ever
                           learn smooth functions loses little.

All perturbations are applied after preprocessing, when every column is on a unit scale.
Validation and test sets are never smoothed: models are always scored on the true labels.
"""

from typing import Dict, Tuple

import numpy as np
import torch

from datasets import TabularTask

__all__ = ["add_noise_features", "rotate_features", "smooth_targets", "smooth_to_fraction"]


def add_noise_features(task: TabularTask, ratio: float, seed: int = 0) -> TabularTask:
    """Append round(ratio * n_features) columns of pure N(0, 1) noise to every split."""
    n_new = int(round(ratio * task.n_features))
    if n_new == 0:
        return task
    rng = np.random.RandomState(seed)
    noise = lambda n: rng.standard_normal((n, n_new))
    return task.with_(
        X_tr=np.hstack([task.X_tr, noise(len(task.X_tr))]),
        X_va=np.hstack([task.X_va, noise(len(task.X_va))]),
        X_te=np.hstack([task.X_te, noise(len(task.X_te))]),
        feature_names=task.feature_names + [f"noise{i}" for i in range(n_new)],
        info={**task.info, "noise_features": n_new},
    )


def random_rotation(d: int, seed: int = 0) -> np.ndarray:
    """Haar-uniform random orthogonal matrix (QR of a Gaussian matrix, sign-corrected)."""
    rng = np.random.RandomState(seed)
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def rotate_features(task: TabularTask, seed: int = 0) -> TabularTask:
    """Apply the same random rotation to every split. Every new column is a blend of all
    the original ones, so "feature j" stops meaning anything, while distances, and
    therefore everything a rotation-invariant learner can see, are unchanged."""
    Q = random_rotation(task.n_features, seed)
    return task.with_(X_tr=task.X_tr @ Q, X_va=task.X_va @ Q, X_te=task.X_te @ Q,
                      feature_names=[f"rot{i}" for i in range(task.n_features)],
                      info={**task.info, "rotated": True})


def _median_nn_distance(X: torch.Tensor, chunk: int = 2048) -> float:
    d_min = []
    for i in range(0, len(X), chunk):
        D = torch.cdist(X[i:i + chunk], X)
        D[torch.arange(D.shape[0]), torch.arange(i, i + D.shape[0])] = float("inf")   # skip self
        d_min.append(D.min(dim=1).values)
    return float(torch.cat(d_min).median())


def smooth_targets(task: TabularTask, bandwidth_mult: float,
                   chunk: int = 1024) -> Tuple[TabularTask, Dict[str, float]]:
    """Replace each TRAINING label by a Gaussian-kernel average of its neighbours' labels.

    Bandwidth h = bandwidth_mult x (median nearest-neighbour distance in the training set),
    so the same multiplier means "a similar neighbourhood size" on datasets with very
    different dimensionality. Classification: kernel-averaged one-hot labels, then argmax
    (a soft nearest-neighbour vote). Regression: kernel-averaged target.

    Returns the smoothed task and {bandwidth, labels_changed} — the fraction of training
    labels the smoothing altered, so the strength is comparable across datasets.
    """
    X = torch.tensor(task.X_tr, dtype=torch.float32)
    h = bandwidth_mult * _median_nn_distance(X)

    if task.task == "regression":
        Y = torch.tensor(task.y_tr, dtype=torch.float32)[:, None]
    else:
        Y = torch.nn.functional.one_hot(torch.as_tensor(task.y_tr).long(),
                                        num_classes=task.n_classes).float()

    out = []
    for i in range(0, len(X), chunk):
        W = torch.exp(-torch.cdist(X[i:i + chunk], X).pow(2) / (2 * h * h))   # includes self
        out.append((W @ Y) / W.sum(dim=1, keepdim=True))
    S = torch.cat(out)

    if task.task == "regression":
        y_new = S[:, 0].numpy().astype(task.y_tr.dtype)
        changed = float(np.mean(np.abs(y_new - task.y_tr) > 1e-6))
    else:
        y_new = S.argmax(dim=1).numpy().astype(task.y_tr.dtype)
        changed = float(np.mean(y_new != task.y_tr))

    info = {"bandwidth": h, "labels_changed": changed}
    return task.with_(y_tr=y_new, info={**task.info, "smoothing": info}), info


def smooth_to_fraction(task: TabularTask, target: float, tol: float = 0.01,
                       lo: float = 0.05, hi: float = 6.0,
                       max_iter: int = 16) -> Tuple[TabularTask, Dict[str, float]]:
    """Smooth with the bandwidth that changes ~`target` of the training labels.

    A fixed bandwidth multiplier means very different things on different datasets (the
    same x1.0 changes 13% of Covertype's labels but 43% of HIGGS's, whose labels are noisy
    even between close neighbours). Fixing the fraction of labels changed instead makes
    "light" and "strong" smoothing comparable across datasets. Bisection on a log scale;
    the fraction changed grows with the bandwidth.
    """
    best = None
    for _ in range(max_iter):
        mid = float(np.sqrt(lo * hi))
        smoothed, info = smooth_targets(task, mid)
        info = {**info, "bandwidth_mult": mid}
        if best is None or abs(info["labels_changed"] - target) < abs(best[1]["labels_changed"] - target):
            best = (smoothed, info)
        if abs(info["labels_changed"] - target) <= tol:
            break
        if info["labels_changed"] < target:
            lo = mid
        else:
            hi = mid
    return best


if __name__ == "__main__":
    from datasets import make_task
    for name in ("covertype", "poker", "higgs"):
        t = make_task(name)
        n = add_noise_features(t, 1.0)
        r = rotate_features(t)
        # a rotation preserves all pairwise distances
        dist_ok = np.allclose(np.linalg.norm(t.X_te[:50] - t.X_te[1:51], axis=1),
                              np.linalg.norm(r.X_te[:50] - r.X_te[1:51], axis=1), atol=1e-8)
        line = (f"{name:9s} noise: {t.n_features} -> {n.n_features} features | "
                f"rotation preserves distances: {dist_ok} | smoothing labels changed:")
        for m in (0.5, 1.0, 2.0, 3.0):
            _, info = smooth_targets(t, m)
            line += f"  x{m}: {info['labels_changed']:.1%}"
        print(line)
