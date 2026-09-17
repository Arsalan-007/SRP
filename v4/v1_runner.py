"""
v1_runner.py  —  run the ORIGINAL V1 model on a TabularTask.

V1 ("MLPs with Attention V1") is the unchanged code vendored in v4/v1/: K shallow MLPs, each
on a random feature subset, feeding a bidirectional Transformer encoder with a [CLS] token,
trained with a CKA diversity penalty. This wrapper only:
  * seeds torch/numpy, picks the device,
  * calls V1's own `train_joint` (V1's training loop, optimiser, early stopping),
  * evaluates the returned model on the FULL test set with the paper's metric
    (V1's internal test score averages per-batch metrics, which is slightly biased).

Hyperparameters are V1's own defaults from tabjoint_vishal_averaged_experiments/hparams.py
(the GRID values used when a parameter is not being swept) plus its STATIC settings.
Note: V1's train_joint hard-codes feature_frac=0.6 and resample_each_epoch=True, and its
resampling re-seeds with the same seed every time, so the subsets are in fact fixed.
"""

import contextlib
import io
import time
from typing import Dict, Optional

import numpy as np
import torch

__all__ = ["V1_DEFAULTS", "run_v1", "default_device"]

V1_DEFAULTS = dict(
    # ensemble of MLPs (hparams.GRID defaults)
    n_mlps=8, emb_dim=128, mlp_hidden_layers=1, mlp_dropout=0.1,
    # training
    epochs=30, batch_size=256, lr_mlp=1e-3, lr_tr=1e-3, lambda_cka=0.4, lambda_div=0.0,
    # transformer head
    d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1,
    # hparams.STATIC
    weight_decay=1e-5, patience=10,
)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _predict(model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(np.ascontiguousarray(X[i:i + batch_size]), dtype=torch.float32, device=device)
        logits, _, _ = model(xb)
        out.append(logits.float().cpu())
    return torch.cat(out).numpy()


def run_v1(task, seed: int, device: Optional[str] = None, hp: Optional[Dict] = None,
           verbose: bool = False, threads: Optional[int] = None) -> Dict:
    """Train V1 once on `task` and return test predictions + metadata."""
    from v1.models import TCfg, TrainCfg
    from v1.training import train_joint

    if threads:
        torch.set_num_threads(threads)
    device = device or default_device()
    hp = {**V1_DEFAULTS, **(hp or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)

    a = task.arrays()
    ydt = np.int64 if task.task in ("binary", "multiclass") else np.float32
    y = {k: a[k].astype(ydt) for k in ("ytr", "yva", "yte")}

    tcfg = TCfg(d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"],
                dim_feedforward=hp["dim_feedforward"], dropout=hp["dropout"])
    cfg = TrainCfg(epochs=hp["epochs"], batch_size=hp["batch_size"], lr_mlp=hp["lr_mlp"],
                   lr_tr=hp["lr_tr"], weight_decay=hp["weight_decay"], patience=hp["patience"],
                   lambda_div=hp["lambda_div"], lambda_cka=hp["lambda_cka"])

    t0 = time.time()
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with sink as log:
        model, scores = train_joint(
            a["Xtr"], y["ytr"], a["Xva"], y["yva"], a["Xte"], y["yte"],
            in_dim=task.n_features, n_mlps=hp["n_mlps"], emb_dim=hp["emb_dim"],
            mlp_hidden_layers=hp["mlp_hidden_layers"], mlp_dropout=hp["mlp_dropout"],
            out_dim=task.output_dim, task=task.task, tcfg=tcfg, cfg=cfg, device=device)
    epochs_run = log.getvalue().count("Epoch ") if log is not None else None

    return {"seed": seed, "device": device,
            "test_pred": _predict(model, a["Xte"], device),
            "val_pred": _predict(model, a["Xva"], device),
            "v1_best_val": scores["best_val"], "epochs_run": epochs_run,
            "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "seconds": time.time() - t0}
