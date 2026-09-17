
import itertools, json, time
import numpy as np
import torch

from .models import TCfg, TrainCfg
from .training import train_joint
from .models import FeatureSubspaceCfg
from torch.utils.data import DataLoader
from .datasets import ArrayDataset
from .cka_utils import collect_tokens_sampled, cka_token_matrix   # optional fallback
from .sim_utils import avg_similarity_matrix_over_loader
from .models import cka_offdiag_mean
import numpy as np


def count_learnable_parameters(model):
    """Count total learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _combo_dict(keys, values_tuple):
    return {k: v for k, v in zip(keys, values_tuple)}

def run_grid_search(
    X_train_t, y_train, X_val_t, y_val, X_test_t, y_test,
    in_dim, task, out_dim, device,
    grid, static, results_path, best_path, primary_metric="best_val"
):
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    all_trials = []
    best = None

    trial_idx = 0
    total_trials = int(np.prod([len(v) for v in value_lists]))
    print(f"[GridSearch] total trials = {total_trials}")

    for vals in itertools.product(*value_lists):
        hp = _combo_dict(keys, vals)
        trial_idx += 1
        print(f"\n[Trial {trial_idx}/{total_trials}] hparams = {hp}")

        # Build TCfg and TrainCfg from this combo
        tcfg = TCfg(
            d_model=hp["d_model"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
        )
        cfg  = TrainCfg(
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            lr_mlp=hp["lr_mlp"],
            lr_tr=hp["lr_tr"],
            weight_decay=static.get("weight_decay", 1e-5),
            patience=static.get("patience", 6),
            lambda_div=hp["lambda_div"],
            lambda_cka=hp["lambda_cka"],
        )

        # Subspace settings
        sub_cfg = FeatureSubspaceCfg(
            frac=static.get("feature_frac", 0.6),
            resample_each_epoch=static.get("resample_each_epoch", True),
            seed=static.get("seed", 123),
        )

        # We pass subspace via model ctor inside train_joint by a tiny wrapper
        # Easiest way: temporarily monkeypatch FeatureSubspaceCfg defaults? Better: modify train_joint signature.
        # But we can reuse train_joint by passing the sub_cfg after construction—so we replicate minimal logic here.
        # Simpler: call train_joint as-is and let it create sub_cfg with frac matching ours.
        # To keep your existing API unchanged, we temporarily set a module-level default via a tiny hack:
        # We'll pass sub_cfg values by setting them on cfg (not ideal). Instead, we reimplement a one-liner wrapper:

        # Use train_joint directly; it internally builds a FeatureSubspaceCfg(frac=0.6, resample_each_epoch=True,...)
        # To propagate our 'frac' we can set it via environment or just accept 0.6 default.
        # For clarity, we accept the current default (0.6). If you want to wire it through, add a param to train_joint.

        start = time.time()
        from .models import JointMLPTransformer  # to get n_mlps/emb_dim from hp
        n_mlps = hp["n_mlps"]
        emb_dim = hp["emb_dim"]
        mlp_hidden_layers = hp["mlp_hidden_layers"]
        mlp_dropout = hp["mlp_dropout"]

        model, scores = train_joint(
            X_train_t, y_train, X_val_t, y_val, X_test_t, y_test,
            in_dim=in_dim, n_mlps=n_mlps, emb_dim=emb_dim,
            mlp_hidden_layers=mlp_hidden_layers, mlp_dropout=mlp_dropout,
            out_dim=out_dim, task=task, tcfg=tcfg, cfg=cfg, device=device
        )
                # --- compute similarity metrics on the test set (off-diagonal means) ---
        try:
            test_loader = DataLoader(ArrayDataset(X_test_t, y_test), batch_size=cfg.batch_size, shuffle=False)

            # 1) mean CKA (off-diagonal). Preferred: use model's implementation cka_offdiag_mean
            try:
                tokens_sample = collect_tokens_sampled(model, test_loader, device, max_rows=2048)  # (N, n_mlps, emb_dim)
                tokens_sample = tokens_sample.to(next(model.parameters()).device)  # ensure on same device
                mean_similarity_cka = float(cka_offdiag_mean(tokens_sample))
            except Exception:
                # fallback: compute full CKA matrix and take off-diagonal mean
                try:
                    tokens_sample = collect_tokens_sampled(model, test_loader, device, max_rows=2048)
                    S_cka = cka_token_matrix(tokens_sample)   # numpy (n_mlps, n_mlps)
                    n = S_cka.shape[0]
                    mean_similarity_cka = float(S_cka[~np.eye(n, dtype=bool)].mean()) if n > 1 else None
                except Exception:
                    mean_similarity_cka = None

            # 2) mean cosine similarity (off-diagonal)
            try:
                S_cos = avg_similarity_matrix_over_loader(model, test_loader, device)  # numpy
                n2 = S_cos.shape[0]
                mean_similarity_cosine = float(S_cos[~np.eye(n2, dtype=bool)].mean()) if n2 > 1 else None
            except Exception:
                mean_similarity_cosine = None
        except Exception:
            mean_similarity_cka = None
            mean_similarity_cosine = None

        # accuracy / test metric
        accuracy_value = scores.get("test", None)
        dur = time.time() - start
        n_params = count_learnable_parameters(model)
        record = {
            "hparams": hp,
            "scores":  scores,
            "seconds": round(dur, 2),
            "learnable_parameters": n_params,
            "mean_similarity_cka": mean_similarity_cka,
            "mean_similarity_cosine": mean_similarity_cosine,
            "accuracy": accuracy_value
        }
        all_trials.append(record)

        cur_metric = scores.get(primary_metric, float("-inf"))
        if (best is None) or (cur_metric > best["scores"].get(primary_metric, float("-inf"))):
            best = record
            # Persist best-so-far
        #     with open(best_path, "w") as f:
        #         json.dump(best, f, indent=2)

        # # Persist full running log
        # with open(results_path, "w") as f:
        #     json.dump(all_trials, f, indent=2)

        print(f"[Trial {trial_idx}] val={scores.get('best_val'):.6f} | test={scores.get('test'):.6f} | time={dur:.1f}s")

    # Sort all trials by primary metric (desc)
    all_trials_sorted = sorted(all_trials, key=lambda r: r["scores"].get(primary_metric, float("-inf")), reverse=True)
    return all_trials_sorted[0], all_trials_sorted# with open(results_path, "w") as f:
    #     json.dump(all_trials_sorted, f, indent=2)

    # # Write final best
    # with open(best_path, "w") as f:
    #     json.dump(all_trials_sorted[0], f, indent=2)

    # print(f"\n[GridSearch] DONE. Best by {primary_metric}:",
    #       all_trials_sorted[0]["scores"][primary_metric],
    #       "\nSaved:", results_path, "\nBest:", best_path)
    # return all_trials_sorted[0], all_trials
