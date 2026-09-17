"""
tuning.py  —  SRP v4

Optuna hyperparameter search, shared by the notebooks.

Neural studies
--------------
Several worker processes share ONE study through a journal file, so a study with N
trials finishes ~N_workers times faster. Every trial:
  * builds a model from the sampled parameters (always model seed 0, so trials compare
    configurations rather than lucky initialisations),
  * trains it with the shared `training.train_model` contract,
  * reports the validation loss after every epoch so the MedianPruner can stop trials
    that are clearly worse than the median at the same epoch,
  * returns the best VALIDATION loss of the final prediction. The test set is never seen.
The chosen configuration is then re-trained on several seeds and evaluated on test.

Search spaces
-------------
`num_learners` (K) is searched jointly with the parameters it interacts with. The best K
depends on how wide each expert is and how many columns it sees, so tuning K with
everything else frozen would just return the best K for the step-3 defaults.

XGBoost
-------
Tuned with the same number of Optuna trials as each neural study, so neither side of the
comparison gets more search effort than the other.
"""

import sys
import time
from typing import Dict, Optional

import numpy as np
import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from attention import AttentionConfig
from embeddings import EmbeddingConfig
from model import PlainMLP, SRPConfig, SRPModel
from training import evaluate, train_model
from weak_learners import WeakLearnerConfig

__all__ = ["NN_ARMS", "STEP3_DEFAULTS", "suggest_params", "build_model", "run_config",
           "create_study", "optuna_worker", "tune_xgb", "make_xgb", "xgb_metrics",
           "load_study"]

NN_ARMS = ("meanpool", "causal")

# The step-3 configurations, expressed in this module's parameter format.
STEP3_DEFAULTS = {
    "meanpool": dict(num_learners=16, hidden_dim=16, embed_dim=32, depth=1, feature_frac=0.7,
                     dropout=0.1, lr=1e-3, weight_decay=1e-4, head_hidden=148, head_depth=2),
    "causal":   dict(num_learners=16, hidden_dim=16, embed_dim=32, depth=1, feature_frac=0.7,
                     dropout=0.1, lr=1e-3, weight_decay=1e-4, num_heads=4, attn_depth=2),
}


# ---------------------------------------------------------------------------
# Search space & model construction
# ---------------------------------------------------------------------------
def suggest_params(trial: optuna.Trial, arm: str, k_range=(2, 64)) -> Dict:
    p = {
        "num_learners": trial.suggest_int("num_learners", k_range[0], k_range[1], log=True),
        "hidden_dim":   trial.suggest_int("hidden_dim", 8, 64, log=True),
        "embed_dim":    trial.suggest_categorical("embed_dim", [16, 32, 64]),
        "depth":        trial.suggest_int("depth", 1, 2),
        "feature_frac": trial.suggest_float("feature_frac", 0.3, 1.0),
        "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
        "lr":           trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    if arm == "meanpool":
        # The head is mean-pool's only aggregation capacity, so its size is searched.
        p["head_hidden"] = trial.suggest_int("head_hidden", 32, 256, log=True)
        p["head_depth"] = trial.suggest_int("head_depth", 1, 2)
    elif arm == "causal":
        p["num_heads"] = trial.suggest_categorical("num_heads", [2, 4])
        p["attn_depth"] = trial.suggest_int("attn_depth", 1, 3)
    else:
        raise ValueError(f"unknown arm {arm!r}; choose from {NN_ARMS}")

    # Per-feature embeddings (embeddings.py) are now part of the architecture, so a real
    # hyperparameter search has to cover them too — searching everything else with the
    # embedding frozen at one setting would just find the best config FOR that setting.
    p["embedding_mode"] = trial.suggest_categorical("embedding_mode", ["none", "linear", "periodic"])
    if p["embedding_mode"] != "none":
        p["d_embedding"] = trial.suggest_categorical("d_embedding", [4, 8, 16])
        if p["embedding_mode"] == "periodic":
            p["n_frequencies"] = trial.suggest_categorical("n_frequencies", [8, 16, 32])
            p["sigma"] = trial.suggest_float("sigma", 0.01, 1.0, log=True)
    return p


def _embedding_config(params: Dict) -> EmbeddingConfig:
    mode = params.get("embedding_mode", "none")
    if mode == "none":
        return EmbeddingConfig(mode="none")
    kw = dict(mode=mode, d_embedding=params["d_embedding"])
    if mode == "periodic":
        kw.update(n_frequencies=params["n_frequencies"], sigma=params["sigma"])
    return EmbeddingConfig(**kw)


def build_model(arm: str, params: Dict, in_dim: int, seed: int, task: str = "regression",
                output_dim: int = 1, batched: bool = True) -> SRPModel:
    """Model for one parameter set. Sets torch.manual_seed(seed) before construction."""
    ens = WeakLearnerConfig(num_learners=params["num_learners"], hidden_dim=params["hidden_dim"],
                            embed_dim=params["embed_dim"], depth=params["depth"],
                            dropout=params["dropout"], feature_frac=params["feature_frac"],
                            seed=seed, batched=batched)
    torch.manual_seed(seed)
    emb = _embedding_config(params)
    if arm == "meanpool":
        cfg = SRPConfig(aggregator="meanpool", head_hidden=params["head_hidden"],
                        head_depth=params["head_depth"], head_dropout=params["dropout"],
                        task=task, output_dim=output_dim, embedding=emb)
        return SRPModel(in_dim, ens, None, cfg)
    if arm == "causal":
        att = AttentionConfig(embed_dim=params["embed_dim"], num_heads=params["num_heads"],
                              attn_depth=params["attn_depth"], dropout=params["dropout"],
                              need_weights=False)
        cfg = SRPConfig(aggregator="causal", readout="last", head_dropout=params["dropout"],
                        task=task, output_dim=output_dim, embedding=emb)
        return SRPModel(in_dim, ens, att, cfg)
    raise ValueError(f"unknown arm {arm!r}; choose from {NN_ARMS}")


def _tensors(arrays: Dict[str, np.ndarray], device=None) -> Dict[str, torch.Tensor]:
    from training import to_tensors
    return to_tensors(arrays, device)


def run_config(arm: str, params: Dict, arrays: Dict[str, np.ndarray], seed: int,
               task: str = "regression", output_dim: int = 1, epochs: int = 200,
               patience: int = 20, threads: Optional[int] = None) -> Dict:
    """Train one configuration on one seed with the standard protocol; val + test metrics."""
    if threads:
        torch.set_num_threads(threads)
    data = _tensors(arrays)
    model = build_model(arm, params, data["Xtr"].shape[1], seed, task, output_dim)
    model, hist = train_model(model, data, seed=seed, epochs=epochs, patience=patience,
                              lr=params["lr"], weight_decay=params["weight_decay"])
    val = evaluate(model, data["Xva"], arrays["yva"], task)
    test = evaluate(model, data["Xte"], arrays["yte"], task)
    return {"arm": arm, "seed": seed, "K": params["num_learners"],
            "n_params": model.num_parameters(),
            **{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()},
            "best_epoch": hist["best_epoch"] + 1, "seconds": hist["seconds"]}


# ---------------------------------------------------------------------------
# Parallel Optuna study
# ---------------------------------------------------------------------------
def _storage(path: str) -> JournalStorage:
    return JournalStorage(JournalFileBackend(str(path)))


def _sampler(seed: int) -> optuna.samplers.TPESampler:
    # constant_liar: running trials count as "bad" so parallel workers spread out
    # instead of all sampling the same promising region.
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True,
                                      n_startup_trials=10)


def _pruner() -> optuna.pruners.MedianPruner:
    return optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=20)


def create_study(study_name: str, storage_path: str) -> optuna.Study:
    return optuna.create_study(study_name=study_name, storage=_storage(storage_path),
                               direction="minimize", sampler=_sampler(0), pruner=_pruner(),
                               load_if_exists=True)


def load_study(study_name: str, storage_path: str) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=_storage(storage_path))


def optuna_worker(arm: str, study_name: str, storage_path: str, n_trials: int, worker_seed: int,
                  arrays: Dict[str, np.ndarray], task: str = "regression", output_dim: int = 1,
                  threads: int = 2, trial_epochs: int = 150, trial_patience: int = 15,
                  project_dir: Optional[str] = None) -> int:
    """Run `n_trials` trials of a shared study inside one worker process."""
    if project_dir and project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    torch.set_num_threads(threads)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=_storage(storage_path),
                              sampler=_sampler(worker_seed), pruner=_pruner())
    data = _tensors(arrays)
    in_dim = data["Xtr"].shape[1]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, arm)
        model = build_model(arm, params, in_dim, seed=0, task=task, output_dim=output_dim)
        trial.set_user_attr("n_params", model.num_parameters())

        def report(epoch, val_loss):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.time()
        model, hist = train_model(model, data, seed=0, epochs=trial_epochs,
                                  patience=trial_patience, lr=params["lr"],
                                  weight_decay=params["weight_decay"], epoch_callback=report)
        trial.set_user_attr("best_epoch", hist["best_epoch"] + 1)
        trial.set_user_attr("seconds", time.time() - t0)
        return hist["best_val"]

    study.optimize(objective, n_trials=n_trials)
    return worker_seed


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def _xgb_suggest(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
    }


def make_xgb(params: Dict, task: str, seed: int = 0, n_jobs: int = -1):
    from xgboost import XGBClassifier, XGBRegressor
    common = dict(tree_method="hist", random_state=seed, n_jobs=n_jobs, **params)
    return XGBRegressor(**common) if task == "regression" else XGBClassifier(**common)


def xgb_metrics(model, X, y, task: str) -> Dict[str, float]:
    from training import classification_metrics, regression_metrics
    if task == "regression":
        return regression_metrics(y, model.predict(X))
    if task == "binary":
        p = model.predict_proba(X)[:, 1]
        logits = np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(1 - p, 1e-12, 1))
        return classification_metrics(y, logits, task)
    return classification_metrics(y, model.predict_proba(X), task)


def tune_xgb(arrays: Dict[str, np.ndarray], task: str, n_trials: int = 40, seed: int = 0,
             n_jobs: int = -1) -> optuna.Study:
    """TPE search scored on the validation set (R² for regression, accuracy otherwise)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    key = "r2" if task == "regression" else "acc"
    ytr = arrays["ytr"] if task == "regression" else arrays["ytr"].astype(int)

    def objective(trial):
        m = make_xgb(_xgb_suggest(trial), task, seed, n_jobs).fit(arrays["Xtr"], ytr)
        return xgb_metrics(m, arrays["Xva"], arrays["yva"], task)[key]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    study.optimize(objective, n_trials=n_trials)
    return study
