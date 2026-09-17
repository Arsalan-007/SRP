
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import SEED, DEVICE, CSV_PATH, TARGET_COL, N_MLPS, EMB_DIM, MLP_HIDDEN_LAYERS, MLP_DROPOUT, TRAIN_KW
from .preprocessing import TabularPreprocessor, PrepConfig, build_label_mapping, apply_mapping
from .models import TCfg, TrainCfg
from .training import train_joint
from .datasets import ArrayDataset
from .sim_utils import avg_similarity_matrix_over_loader, plot_sim_matrix
from .cka_utils import collect_tokens_sampled, cka_token_matrix, plot_matrix
from .metrics_helpers import evaluate_on_loader
from .stats_utils import offdiag_stats

def run():
    # LOAD YOUR DATA
    df = pd.read_csv(CSV_PATH)
    df = df.iloc[:-1]
    if 'PassengerId' in df.columns:
        df = df.drop('PassengerId', axis=1)
    y_raw = df[TARGET_COL].values
    X_raw = df.drop(columns=[TARGET_COL])
    print(y_raw)

    is_reg = (y_raw.dtype.kind in {"f"} and np.unique(y_raw).size > 10)
    if is_reg:
        X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=SEED)
    else:
        X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw)

    if is_reg:
        X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(X_tr, y_tr_raw, test_size=0.2, random_state=SEED)
    else:
        X_tr, X_va, y_tr_raw, y_va_raw = train_test_split(X_tr, y_tr_raw, test_size=0.2, random_state=SEED, stratify=y_tr_raw)

    # --- configure & fit on TRAIN only ---
    prep = TabularPreprocessor(
        PrepConfig(
            impute_num="mean",
            impute_cat="most_frequent",
            scale="standard",
            noise_std=0.0,
            cardinality_threshold=50,
            rare_threshold_pct=0.05,
            cat_encoding="onehot",
            other_token="__OTHER__",
            missing_token="__MISSING__",
        )
    ).fit(X_tr)

    # --- transform all splits using the learned mapping ---
    X_train_t = prep.transform(X_tr)
    X_val_t   = prep.transform(X_va)
    X_test_t  = prep.transform(X_te)

    # --- labels (unchanged) ---
    info = build_label_mapping(y_tr_raw)
    task, out_dim = info["task"], info["out_dim"]
    y_train = apply_mapping(y_tr_raw, info)
    y_val   = apply_mapping(y_va_raw, info)
    y_test  = apply_mapping(y_te_raw, info)

    in_dim = X_train_t.shape[1]
    print(task, out_dim, in_dim, X_train_t.shape, X_val_t.shape, X_test_t.shape)
    print("Dropped high-card categoricals:", prep.cat_cols_dropped)
    print("Kept categoricals:", prep.cat_cols_kept)
    print("Numeric cols:", prep.num_cols)

    # Model / Train configs
    tcfg = TCfg(d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1)
    cfg  = TrainCfg(**TRAIN_KW)

    model, scores = train_joint(
        X_train_t, y_train, X_val_t, y_val, X_test_t, y_test,
        in_dim=in_dim, n_mlps=N_MLPS, emb_dim=EMB_DIM,
        mlp_hidden_layers=MLP_HIDDEN_LAYERS, mlp_dropout=MLP_DROPOUT,
        out_dim=out_dim, task=task, tcfg=tcfg, cfg=cfg, device=DEVICE
    )

    # Similarity
    val_loader = DataLoader(ArrayDataset(X_val_t, y_val), batch_size=1024, shuffle=False)
    S_val = avg_similarity_matrix_over_loader(model, val_loader, device=DEVICE)
    print(offdiag_stats(S_val))
    plot_sim_matrix(S_val, title="Val avg cosine similarity (end-to-end)")

    # CKA
    val_loader = DataLoader(ArrayDataset(X_val_t, y_val), batch_size=1024, shuffle=False)
    val_tokens = collect_tokens_sampled(model, val_loader, device=DEVICE, max_rows=4096)
    S_cka = cka_token_matrix(val_tokens)
    print("CKA off-diagonal stats:", offdiag_stats(S_cka))
    plot_matrix(S_cka, "CKA similarity across MLP tokens (val)")

    # Metrics
    val_loader  = DataLoader(ArrayDataset(X_val_t,  y_val),  batch_size=1024, shuffle=False)
    test_loader = DataLoader(ArrayDataset(X_test_t, y_test), batch_size=1024, shuffle=False)
    val_metrics  = evaluate_on_loader(model, val_loader,  DEVICE, task, threshold=0.5, return_report=True)
    test_metrics = evaluate_on_loader(model, test_loader, DEVICE, task, threshold=0.5, return_report=True)

    print("VAL:",  {k: v for k, v in val_metrics.items() if k not in ("confusion_matrix","classification_report")})
    print("Confusion Matrix (val):\n", val_metrics["confusion_matrix"])
    if "classification_report" in val_metrics:
        print(val_metrics["classification_report"])

    print("TEST:", {k: v for k, v in test_metrics.items() if k not in ("confusion_matrix","classification_report")})
    print("Confusion Matrix (test):\n", test_metrics["confusion_matrix"])

    # 1) Print the indices (and names) per MLP
    def show_feature_subsets(model, feature_names=None, max_names=20):
        for i, idx in enumerate(model.feature_subsets):
            if feature_names is None:
                print(f"MLP {i}: {len(idx)} cols → {idx[:min(10, len(idx))]} ...")
            else:
                names = [feature_names[j] for j in idx]
                preview = ", ".join(names[:max_names]) + (" ..." if len(names) > max_names else "")
                print(f"MLP {i}: {len(idx)} cols → {preview}")

    show_feature_subsets(model, getattr(prep, "feature_names_", None))

if __name__ == "__main__":
    run()
