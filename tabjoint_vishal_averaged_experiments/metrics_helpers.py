
import numpy as np, torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

@torch.no_grad()
def collect_logits_and_labels(model, loader, device, task):
    model.eval()
    logits_list, y_list = [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits, _, _ = model(xb)
        logits_list.append(logits.detach().cpu())
        y_list.append(yb.detach().cpu())
    logits = torch.cat(logits_list, 0).numpy()
    y_true = torch.cat(y_list, 0).numpy()
    return logits, y_true

def evaluate_on_loader(model, loader, device, task, threshold=0.5, return_report=False):
    """
    Returns a dict of metrics.
    - binary: accuracy, precision, recall, f1, roc_auc, pr_auc, confusion_matrix
    - multiclass: accuracy, precision/recall/f1 (macro & weighted), confusion_matrix
    - regression: rmse, mae, r2
    """
    logits, y_true = collect_logits_and_labels(model, loader, device, task)

    if task == "binary":
        y_prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))         # sigmoid
        y_pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        roc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)              # PR-AUC
        cm  = confusion_matrix(y_true, y_pred)
        out = {
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "roc_auc": roc, "pr_auc": ap, "confusion_matrix": cm
        }
        if return_report:
            out["classification_report"] = classification_report(y_true, y_pred, digits=4, zero_division=0)
        return out

    if task == "multiclass":
        y_pred = logits.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_w, r_w, f1_w, _       = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        cm  = confusion_matrix(y_true, y_pred)
        out = {
            "accuracy": acc,
            "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f1_mac,
            "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
            "confusion_matrix": cm
        }
        if return_report:
            out["classification_report"] = classification_report(y_true, y_pred, digits=4, zero_division=0)
        return out

    # regression
    y_pred = logits.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred)))
    mae  = float(mean_absolute_error(y_true.reshape(-1), y_pred))
    r2   = float(r2_score(y_true.reshape(-1), y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}
