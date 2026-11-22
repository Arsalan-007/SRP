"""
cli.py  —  SRP v4

Run one experiment without a notebook.

The 19 notebooks in this directory are the experiment RECORD: each one asks a specific
question, fixes its decision rule before running, and keeps its output. They are not the
entry point. This is — so that reproducing a number from the paper is one command rather
than a Jupyter session:

    python cli.py --dataset CA --arm meanpool --embedding periodic --seeds 5

Every run writes the same `report.json` that `results.py` defines and the notebooks emit,
so CLI runs and notebook runs land in one results tree and rank together.

    python cli.py --list                      # datasets, arms and embedding modes
    python cli.py --dataset CA --arm causal --seeds 3 --epochs 50
    python cli.py --dataset HI --arm conf_dynamic --out results/my_run

Self-test:  python cli.py --self-test
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import benchmarks
import benchmarks_rtdl
from attention import AttentionConfig
from embeddings import EMBEDDING_MODES, EmbeddingConfig
from model import ARM_NAMES, make_arm
from results import save_report
from training import evaluate, get_device, to_tensors, train_model
from weak_learners import WeakLearnerConfig, default_ens

__all__ = ["LOADERS", "run_arm", "main"]

# Which module owns which dataset key. The two benchmark suites use different published
# splits and must not be confused, so the mapping is explicit rather than try/except.
LOADERS = {**{k: benchmarks_rtdl for k in benchmarks_rtdl.BENCHMARKS},
           **{k: benchmarks for k in benchmarks.BENCHMARKS}}

DEFAULTS = default_ens()      # single source of truth: weak_learners.default_ens()


def run_arm(dataset: str, arm: str, seeds=(0,), embedding: str = "none",
            epochs: int = 400, patience: int = 20, lr: float = 1e-3,
            batch_size: int = 512, weight_decay: float = 1e-4,
            subsample: int = 0, out_dir=None, device=None,
            expert_loss_beta: float = 0.0,
            quiet: bool = False, **ens_overrides) -> Dict:
    """Train `arm` on `dataset` once per seed. Returns the per-seed scores and the summary.

    The score is the dataset's own metric: RMSE in ORIGINAL units for regression (rescaled
    by y_std, like the published numbers), accuracy otherwise.
    """
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(LOADERS)}")
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARM_NAMES}")
    if embedding not in EMBEDDING_MODES:
        raise ValueError(f"unknown embedding {embedding!r}; choose from {EMBEDDING_MODES}")

    loader = LOADERS[dataset]
    data_obj = loader.load_benchmark(dataset)
    arrays = data_obj.arrays()
    if subsample:
        n = subsample
        arrays = {"Xtr": arrays["Xtr"][:n], "ytr": arrays["ytr"][:n],
                  "Xva": arrays["Xva"][:n // 2], "yva": arrays["yva"][:n // 2],
                  "Xte": arrays["Xte"][:n], "yte": arrays["yte"][:n]}

    dev = torch.device(device) if device else get_device()
    tensors = to_tensors(arrays, str(dev))
    higher_is_better = loader.higher_is_better(dataset)
    metric = "rmse" if data_obj.task == "regression" else "acc"
    ens_kw = {**DEFAULTS, **ens_overrides}
    emb_kw = dict(mode=embedding, d_embedding=8, n_frequencies=16, sigma=0.1)
    train_kw = dict(epochs=epochs, patience=patience, lr=lr,
                    batch_size=batch_size, weight_decay=weight_decay)

    if not quiet:
        print(f"{data_obj.label} | arm={arm} | embedding={embedding} | {len(seeds)} seed(s) "
              f"| device={dev}")

    rows = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = make_arm(arm, data_obj.n_features,
                         WeakLearnerConfig(**ens_kw, seed=seed, batched=True),
                         AttentionConfig(embed_dim=ens_kw["embed_dim"], num_heads=4, attn_depth=2),
                         task=data_obj.task, output_dim=data_obj.output_dim,
                         embedding=EmbeddingConfig(**emb_kw),
                         expert_loss_beta=expert_loss_beta)
        t0 = time.time()
        model, hist = train_model(model, tensors, seed=seed, **train_kw)
        test = evaluate(model, tensors["Xte"], arrays["yte"], data_obj.task)
        score = test["rmse"] * data_obj.y_std if data_obj.task == "regression" else test["acc"]
        row = {"seed": seed, "score": float(score), "best_epoch": hist["best_epoch"] + 1,
               "epochs_run": hist["epochs_run"], "seconds": time.time() - t0,
               "n_parameters": model.num_parameters()}
        rows.append(row)

        if out_dir is not None:
            save_report(Path(out_dir) / dataset / arm / f"seed{seed}",
                        dataset=dataset, label=data_obj.label, arm=arm, seed=seed,
                        source="reproduced", source_ref="cli.py"
                        + (" [SUBSAMPLED]" if subsample else ""),
                        config={"ens": ens_kw, "emb": emb_kw, "train": train_kw,
                                "expert_loss_beta": expert_loss_beta},
                        n_parameters=row["n_parameters"],
                        metrics={"test": {metric: row["score"]}},
                        champion_metric=f"test_{metric}", champion_value=row["score"],
                        higher_is_better=higher_is_better,
                        time_seconds=row["seconds"], checkpoint=None)
        if not quiet:
            print(f"  seed {seed}  {metric} {row['score']:.4f}  "
                  f"best epoch {row['best_epoch']:3d}/{row['epochs_run']:3d}  "
                  f"({row['seconds']/60:.1f} min)")

    scores = np.array([r["score"] for r in rows])
    # A best epoch inside the last `patience` epochs means early stopping never fired —
    # the run was still improving at the cap, so its score understates the arm.
    undertrained = [r["seed"] for r in rows if r["best_epoch"] > epochs - patience]
    summary = {"dataset": dataset, "label": data_obj.label, "arm": arm,
               "embedding": embedding, "metric": metric,
               "expert_loss_beta": expert_loss_beta,
               "higher_is_better": higher_is_better,
               "mean": float(scores.mean()), "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
               "n_parameters": rows[0]["n_parameters"], "runs": rows,
               "undertrained_seeds": undertrained}
    if not quiet:
        print(f"  {metric} {summary['mean']:.4f} +/- {summary['std']:.4f} "
              f"over {len(seeds)} seed(s)   ({summary['n_parameters']:,} params)")
        if undertrained:
            print(f"  WARNING: seed(s) {undertrained} hit the {epochs}-epoch cap — raise --epochs")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run one SRP arm on one dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dataset", help=f"one of {sorted(LOADERS)}")
    ap.add_argument("--arm", default="meanpool", help=f"one of {list(ARM_NAMES)}")
    ap.add_argument("--embedding", default="none", help=f"one of {list(EMBEDDING_MODES)}")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds, 0..N-1")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-learners", type=int, default=DEFAULTS["num_learners"])
    ap.add_argument("--feature-frac", type=float, default=DEFAULTS["feature_frac"])
    ap.add_argument("--embed-dim", type=int, default=DEFAULTS["embed_dim"])
    ap.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"],
                    help="width of each expert. With --shared this is nearly free, which is "
                         "the whole point of weight sharing (TabM tunes 64..1024).")
    ap.add_argument("--shared", action="store_true",
                    help="TabM-style BatchEnsemble: K members over one shared weight per "
                         "layer instead of K independent MLPs. Default off (unchanged).")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="expert_loss_beta: weight on the per-expert loss term. "
                         "Required (>0) for --arm expert_beta / conf_beta, ignored by others.")
    ap.add_argument("--subsample", type=int, default=0, help="rows per split, for a smoke test")
    ap.add_argument("--out", default=None, help="results dir for report.json (default: none)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--list", action="store_true", help="list datasets, arms and embeddings")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        print("datasets  :", ", ".join(sorted(LOADERS)))
        print("arms      :", ", ".join(ARM_NAMES))
        print("embeddings:", ", ".join(EMBEDDING_MODES))
        return 0
    if a.self_test:
        return _self_test()
    if not a.dataset:
        ap.error("--dataset is required (or use --list / --self-test)")

    summary = run_arm(a.dataset, a.arm, seeds=tuple(range(a.seeds)), embedding=a.embedding,
                      epochs=a.epochs, patience=a.patience, lr=a.lr, batch_size=a.batch_size,
                      weight_decay=a.weight_decay, subsample=a.subsample, out_dir=a.out,
                      device=a.device, quiet=a.json,
                      num_learners=a.num_learners, feature_frac=a.feature_frac,
                      embed_dim=a.embed_dim, hidden_dim=a.hidden_dim, shared=a.shared,
                      expert_loss_beta=a.beta)
    if a.json:
        print(json.dumps(summary, indent=2))
    return 0


def _self_test() -> int:
    """Argument plumbing and routing only — no training, so this stays fast."""
    ok = []
    ok.append(("every dataset key routes to exactly one loader",
               set(benchmarks.BENCHMARKS) & set(benchmarks_rtdl.BENCHMARKS) == set()
               and len(LOADERS) == len(benchmarks.BENCHMARKS) + len(benchmarks_rtdl.BENCHMARKS)))
    for bad, kw in (("dataset", dict(dataset="NOPE", arm="meanpool")),
                    ("arm", dict(dataset="CA", arm="nope")),
                    ("embedding", dict(dataset="CA", arm="meanpool", embedding="nope"))):
        try:
            run_arm(**kw, seeds=(0,), epochs=1, quiet=True); raised = False
        except ValueError:
            raised = True
        except Exception:
            raised = False
        ok.append((f"an unknown {bad} raises ValueError before any training starts", raised))
    # A beta arm without --beta must fail loudly here, not train the wrong loss branch
    # for an hour on the GPU and report it under the wrong arm name.
    try:
        run_arm(dataset="CA", arm="expert_beta", seeds=(0,), epochs=1, quiet=True)
        beta_guarded = False
    except ValueError:
        beta_guarded = True
    except Exception:
        beta_guarded = False
    ok.append(("--arm expert_beta without --beta raises ValueError before training", beta_guarded))
    ok.append(("--list exits 0", main(["--list"]) == 0))
    for msg, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {msg}")
    return 0 if all(p for _, p in ok) else 1


if __name__ == "__main__":
    sys.exit(main())
