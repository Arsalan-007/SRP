"""
training.py  —  SRP v4

One training and evaluation contract for every neural model in the project.

v1, v2 and v3 each had their own training loop, and that is how their evaluation
drifted apart (v2 trained with one aggregation and reported another; model selection
and reporting used different numbers). From here on every notebook imports this file.

The contract
------------
* Objective     : `model.loss(out, y)` if the model defines it (e.g. deep supervision),
                  otherwise MSE on the prediction.
* Selection     : early stopping on `model.criterion(pred, y)` on the VALIDATION set —
                  the loss of the FINAL prediction only, identical for every arm.
                  A deep-supervision model is never selected on its weighted objective.
* Checkpoint    : the best validation epoch is restored before returning.
* Batch order   : drawn from a generator seeded per run, so arms trained with the same
                  seed see the same batches in the same order (paired comparisons).
* Callback      : optional `epoch_callback(epoch, val_loss)` after every epoch. Used by
                  Optuna to report intermediate values and prune hopeless trials; an
                  exception raised inside it (e.g. optuna.TrialPruned) propagates out.
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = ["train_model", "predict", "regression_metrics", "classification_metrics", "evaluate",
           "get_device", "to_tensors"]


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Where to train. Priority: SRP_DEVICE env var, then `prefer`, then CUDA if present, else CPU.

    On the GPU server, `export CUDA_VISIBLE_DEVICES=0` picks the card and this picks CUDA.
    Set SRP_DEVICE=cpu to force CPU even on a GPU machine (useful for a quick check).
    """
    import os
    name = os.environ.get("SRP_DEVICE") or prefer or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_tensors(arrays: Dict[str, np.ndarray], device=None) -> Dict[str, torch.Tensor]:
    """numpy splits -> tensors on the training device.

    np.ascontiguousarray() copies read-only arrays (joblib hands workers memory-mapped, read-only
    copies), which silences torch's non-writable-array warning and keeps the data contiguous.
    """
    dev = torch.device(device) if device is not None else get_device()
    return {k: torch.as_tensor(np.ascontiguousarray(v)).to(dev) for k, v in arrays.items()}


def _pred(out):
    return out["pred"] if isinstance(out, dict) else out


@torch.no_grad()
def _eval_loss(model, X, y, criterion, batch_size: int) -> float:
    """Mean loss over a split, computed in chunks.

    A single forward pass over the whole validation split is what ran a 16 GB GPU out of
    memory: with per-feature embeddings, 32k rows expand into an intermediate of
    rows x learners x features x d_embedding. Chunking is numerically identical for
    mean-reduction losses (we weight each chunk by its size) and bounds peak memory.
    """
    total, n = 0.0, len(X)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        total += criterion(_pred(model(xb)), yb).item() * xb.shape[0]
    return total / max(1, n)


def train_model(model: torch.nn.Module, data: Dict[str, torch.Tensor], *,
                epochs: int = 200, lr: float = 1e-3, batch_size: int = 256,
                weight_decay: float = 1e-4, patience: int = 20, l1_lambda: float = 0.0,
                seed: int = 0, verbose: bool = False, epoch_callback=None,
                eval_batch_size: int = 4096):
    """Train with early stopping. `data` holds Xtr, ytr, Xva, yva tensors on the target device.

    Returns (model, history) where history has per-epoch train objective and
    validation criterion, the best epoch, epochs run, and wall-clock seconds.
    """
    device = data["Xtr"].device
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = getattr(model, "criterion", lambda p, y: F.mse_loss(p, y))
    objective = getattr(model, "loss", None)

    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    n = Xtr.shape[0]
    gen = torch.Generator().manual_seed(seed)

    best_val, best_state, best_epoch, waited = float("inf"), None, -1, 0
    hist = {"train": [], "val": []}
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = objective(out, yb) if objective is not None else criterion(_pred(out), yb)
            if l1_lambda and hasattr(model, "ensemble"):
                loss = loss + model.ensemble.weight_penalty(l1_lambda=l1_lambda)
            loss.backward()
            opt.step()
            running += loss.item() * idx.numel()

        model.eval()
        val = _eval_loss(model, Xva, yva, criterion, eval_batch_size)
        hist["train"].append(running / n)
        hist["val"].append(val)
        if epoch_callback is not None:
            epoch_callback(ep, val)

        if val < best_val - 1e-6:
            best_val, best_epoch, waited = val, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
        if verbose and (ep + 1) % 25 == 0:
            print(f"    epoch {ep + 1:3d} | train {hist['train'][-1]:.4f} | val {val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    hist.update(best_val=best_val, best_epoch=best_epoch,
                epochs_run=len(hist["val"]), seconds=time.time() - t0)
    return model, hist


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
    """Final predictions as a numpy array, in eval mode."""
    model.eval()
    parts = [_pred(model(X[i:i + batch_size])).cpu() for i in range(0, X.shape[0], batch_size)]
    return torch.cat(parts).numpy()


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, logits, task: str) -> Dict[str, float]:
    """Accuracy (primary) and balanced accuracy from raw model outputs (logits)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    logits = np.asarray(logits)
    if task == "binary":
        pred = (logits.reshape(-1) > 0).astype(int)          # logit > 0  <=>  p > 0.5
    else:
        pred = logits.argmax(axis=1)
    return {"acc": float(accuracy_score(y_true, pred)),
            "bal_acc": float(balanced_accuracy_score(y_true, pred))}


def evaluate(model: torch.nn.Module, X: torch.Tensor, y_true, task: str) -> Dict[str, float]:
    """Task-appropriate metrics for a trained neural model."""
    out = predict(model, X)
    return regression_metrics(y_true, out) if task == "regression" \
        else classification_metrics(y_true, out, task)
