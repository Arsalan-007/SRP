"""
hetero.py  —  SRP v4, notebook 20

Heterogeneous weak learners + validation-based greedy ensemble selection.

Inspired by TabPack (Gorishniy, Kotelnikov, Rubachev & Babenko, "TabPack: Efficient
Hyperparameter Ensembles for Tabular Deep Learning", ICML 2026, arXiv:2607.05380), adapted
to v4's weak-learner ensemble. Kept as its own module, like benchmarks_rtdl.py, so nothing
here can disturb the already-validated model.py / training.py pipeline.

What changes relative to today's v4 `meanpool` arm
--------------------------------------------------
                     meanpool (today)                    hetero_select (this file)
  expert settings    one fixed config for all K          drawn at random PER EXPERT
  embedding          one shared PLR for all experts      each expert owns its PLR (own sigma)
  head               one shared head on the mean         each expert owns its head
  training           joint: one loss on the ensemble     independent: each expert its own loss,
                                                         its own lr / weight decay
  combining          average all K embeddings            greedy selection on VALIDATION data,
                                                         rebuilt every epoch, then average
  early stopping     on the model's val loss             experts that stall are retired; training
                                                         stops when the ENSEMBLE stops improving

What does NOT change: K=16, hidden_dim=16, embed_dim=32, feature_frac=0.7, PLR width
(d_embedding=8, n_frequencies=16), batch size, epoch cap, and — for a given seed — the
exact feature subsets and batch order of the control. Width is deliberately NOT varied:
TabPack's own "DiverseWidths" ablation found it does not help.

The `homo_select` control uses this same machinery with every expert at today's fixed
settings, so `hetero_select - homo_select` isolates the hyperparameter diversity itself, and
`homo_select - meanpool` isolates independent training + selection.

Why validation-based selection is a new test, not a repeat of notebooks 15-19
----------------------------------------------------------------------------
Every aggregator that failed before (attention, softmax / sigmoid / Gumbel confidence) learned
its weights on TRAINING data, jointly with the experts — the setting in which learner
collusion (Jeffares et al., NeurIPS 2023) makes apparent diversity fail to generalise. Here
the experts never see each other during training, and membership is chosen on held-out
validation data the experts were not fitted to.

Self-test:  python hetero.py
"""

import copy
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import EmbeddingConfig, FeatureEmbedding
from model import PredictionHead
from weak_learners import ACTIVATIONS, make_feature_subsets

__all__ = ["HeteroSpace", "FIXED_HPARAMS", "sample_hparams", "HeteroExpert", "greedy_select",
           "train_hetero", "to_prob", "prob_loss", "prob_to_output"]

TASKS = ("regression", "binary", "multiclass")
_EPS = 1e-7


# ---------------------------------------------------------------------------
# The per-expert search space
# ---------------------------------------------------------------------------
@dataclass
class HeteroSpace:
    """Ranges each expert's settings are drawn from — once, before training, never tuned.

    depth, dropout, sigma: TabPack's Table 7 ranges (dropout: 0 with prob `dropout_zero_prob`,
        else Uniform; the paper does not state that probability, 0.5 is our choice).
    lr: TabPack's AdamW range.
    weight_decay: TabPack samples LogUniform[1e-3, 1] but does not say which optimiser it
        applies to (it also uses Muon). v4's validated default, 1e-4, would sit OUTSIDE that
        range, so we use LogUniform[1e-5, 1e-1]: it contains v4's default and overlaps the
        lower part of TabPack's range.
    """
    depth: tuple = (1, 3)                 # UniformInt, inclusive
    dropout_zero_prob: float = 0.5
    dropout: tuple = (0.0, 0.5)           # Uniform
    lr: tuple = (1e-4, 5e-3)              # LogUniform
    weight_decay: tuple = (1e-5, 1e-1)    # LogUniform
    sigma: tuple = (1e-2, 10.0)           # LogUniform — PLR frequency scale


# Today's v4 settings (notebooks 15-19): the `homo_select` arm gives every expert exactly these.
FIXED_HPARAMS = dict(depth=1, dropout=0.1, lr=1e-3, weight_decay=1e-4, sigma=0.1)


def sample_hparams(num_experts: int, seed: int = 0, homogeneous: bool = False,
                   space: Optional[HeteroSpace] = None) -> List[Dict]:
    """One settings dict per expert. `homogeneous=True` -> every expert gets FIXED_HPARAMS."""
    if homogeneous:
        return [dict(FIXED_HPARAMS) for _ in range(num_experts)]
    space = space or HeteroSpace()
    rng = np.random.RandomState(seed)

    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    out = []
    for _ in range(num_experts):
        depth = int(rng.randint(space.depth[0], space.depth[1] + 1))
        dropout = 0.0 if rng.rand() < space.dropout_zero_prob else float(rng.uniform(*space.dropout))
        out.append(dict(depth=depth, dropout=dropout, lr=logu(*space.lr),
                        weight_decay=logu(*space.weight_decay), sigma=logu(*space.sigma)))
    return out


# ---------------------------------------------------------------------------
# One expert = a complete small model
# ---------------------------------------------------------------------------
class HeteroExpert(nn.Module):
    """Own column subset -> own PLR embedding -> own MLP body -> own prediction head.

    The body is the same shape as weak_learners.WeakLearnerMLP (Linear->act->Dropout x depth,
    then Linear to embed_dim) and the head is model.PredictionHead with step-1 defaults, so at
    FIXED_HPARAMS an expert is exactly "one v4 expert with its own embedding and head".
    """

    def __init__(self, feat_idx: Sequence[int], output_dim: int, depth: int, dropout: float,
                 sigma: float, hidden_dim: int = 16, embed_dim: int = 32, d_embedding: int = 8,
                 n_frequencies: int = 16, activation: str = "gelu", head_dropout: float = 0.1):
        super().__init__()
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        self.register_buffer("feat_idx", torch.from_numpy(feat_idx), persistent=True)
        k = int(feat_idx.size)
        self.embedding = FeatureEmbedding(k, EmbeddingConfig(
            mode="periodic", d_embedding=d_embedding, n_frequencies=n_frequencies, sigma=sigma))
        act = ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        d = k * d_embedding
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, embed_dim))
        self.body = nn.Sequential(*layers)
        self.head = PredictionHead(embed_dim, None, 1, output_dim, head_dropout)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x.index_select(1, self.feat_idx)).flatten(1)
        y = self.head(self.body(h))
        return y.squeeze(-1) if self.output_dim == 1 else y

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Prediction space: experts are averaged as values (regression) or PROBABILITIES
# ---------------------------------------------------------------------------
def to_prob(out: torch.Tensor, task: str) -> torch.Tensor:
    out = out.float()
    if task == "regression":
        return out
    if task == "binary":
        return torch.sigmoid(out)
    return out.softmax(dim=-1)


def prob_loss(p: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    """Loss of an (averaged) prediction in probability space: MSE / log-loss / NLL.
    For a single expert this equals the training criterion, so expert and ensemble scores
    are on one scale."""
    if task == "regression":
        return F.mse_loss(p, y.float())
    if task == "binary":
        return F.binary_cross_entropy(p.clamp(_EPS, 1 - _EPS), y.float())
    return F.nll_loss(p.clamp_min(_EPS).log(), y.long())


def prob_to_output(p: torch.Tensor, task: str) -> torch.Tensor:
    """Back to what training.classification_metrics expects: a logit (binary) or log-probs
    (multiclass). Regression passes through."""
    if task == "regression":
        return p
    if task == "binary":
        p = p.clamp(_EPS, 1 - _EPS)
        return torch.log(p) - torch.log1p(-p)
    return p.clamp_min(_EPS).log()


def _criterion(out: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "regression":
        return F.mse_loss(out.float(), y.float())
    if task == "binary":
        return F.binary_cross_entropy_with_logits(out.float(), y.float())
    return F.cross_entropy(out, y.long())


@torch.no_grad()
def _predict_prob(expert: nn.Module, X: torch.Tensor, task: str, batch_size: int) -> torch.Tensor:
    was = expert.training
    expert.eval()
    parts = [to_prob(expert(X[i:i + batch_size]), task) for i in range(0, X.shape[0], batch_size)]
    if was:
        expert.train()
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# Greedy ensemble selection (Caruana et al., 2004; TabPack Section 3.5)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_select(cands: List[torch.Tensor], y: torch.Tensor, task: str,
                  max_size: int) -> (List[int], float):
    """Build an ensemble from scratch: start from the best single candidate, then keep adding
    whichever candidate most improves the AVERAGED prediction's validation loss. Each candidate
    is used at most once per call (TabPack: "without replacement"); the same expert can still
    appear twice through snapshots from different epochs. Stops at `max_size` or when nothing
    improves. By construction the result is never worse than the best single candidate."""
    singles = [prob_loss(c, y, task).item() for c in cands]
    first = int(np.argmin(singles))
    chosen, total, best = [first], cands[first].clone(), singles[first]
    remaining = set(range(len(cands))) - {first}
    while len(chosen) < max_size and remaining:
        best_j, best_l = None, best
        for j in sorted(remaining):
            l = prob_loss((total + cands[j]) / (len(chosen) + 1), y, task).item()
            if l < best_l - 1e-9:
                best_j, best_l = j, l
        if best_j is None:
            break
        chosen.append(best_j)
        total += cands[best_j]
        best = best_l
        remaining.discard(best_j)
    return chosen, best


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_hetero(data: Dict[str, torch.Tensor], *, task: str, output_dim: int, n_features: int,
                 hparams: List[Dict], feature_frac: float = 0.7, seed: int = 0,
                 epochs: int = 400, batch_size: int = 512, ens_patience: int = 20,
                 expert_patience: int = 10, max_members: Optional[int] = None,
                 hidden_dim: int = 16, embed_dim: int = 32, d_embedding: int = 8,
                 n_frequencies: int = 16, activation: str = "gelu",
                 eval_batch_size: int = 4096, verbose: bool = False,
                 log: Optional[Callable[[str], None]] = None) -> Dict:
    """Train K independent experts and select an ensemble on validation data every epoch.

    `data` holds Xtr, ytr, Xva, yva, Xte tensors on the target device (training.to_tensors).
    Patience: the ENSEMBLE uses `ens_patience` (set equal to the control's patience, 20); each
    expert is retired after `expert_patience` epochs without improving its own validation
    loss (TabPack uses a 1:2 ratio between the two). Retired experts stop training, but their
    earlier snapshots stay eligible while they remain in the ensemble.

    Returns test predictions of the selected ensemble AND of the plain average of all K
    experts at the same epoch (the "no selection" diagnostic), plus selection diagnostics.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    K = len(hparams)
    max_members = max_members or K
    device = data["Xtr"].device
    say = log or print

    # Same subset draw as WeakLearnerEnsemble(WeakLearnerConfig(..., seed=seed)) -> the control.
    subsets = make_feature_subsets(n_features, K, feature_frac=feature_frac, seed=seed,
                                   ensure_coverage=True)
    torch.manual_seed(seed)
    experts = [HeteroExpert(subsets[i], output_dim, hp["depth"], hp["dropout"], hp["sigma"],
                            hidden_dim=hidden_dim, embed_dim=embed_dim, d_embedding=d_embedding,
                            n_frequencies=n_frequencies, activation=activation).to(device)
               for i, hp in enumerate(hparams)]
    # One parameter group per expert: its own learning rate and weight decay. AdamW skips
    # parameters whose grad is None, so a retired expert (excluded from the loss) is frozen.
    opt = torch.optim.AdamW([{"params": e.parameters(), "lr": hp["lr"],
                              "weight_decay": hp["weight_decay"]}
                             for e, hp in zip(experts, hparams)])

    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    n = Xtr.shape[0]
    gen = torch.Generator().manual_seed(seed)      # same batch order as training.train_model

    alive = [True] * K
    exp_best, exp_wait = [float("inf")] * K, [0] * K
    cur_val: List[Optional[torch.Tensor]] = [None] * K
    prev_members: List[Dict] = []                  # records: {expert, epoch, val, module}
    best_ens, best_epoch, ens_wait = float("inf"), -1, 0
    best_members: List[Dict] = []
    best_all: List[nn.Module] = []
    hist = {"ens_val": [], "all_val": [], "best_single_val": [], "alive": [], "ens_size": []}
    t0 = time.time()

    for ep in range(epochs):
        # ---- train every live expert on its OWN loss (summed -> independent gradients)
        live = [i for i in range(K) if alive[i]]
        for i in live:
            experts[i].train()
        perm = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            loss = sum(_criterion(experts[i](xb), yb, task) for i in live)
            loss.backward()
            opt.step()

        # ---- validate live experts, retire the ones that stalled
        trained_now = []
        for i in live:
            cur_val[i] = _predict_prob(experts[i], Xva, task, eval_batch_size)
            v = prob_loss(cur_val[i], yva, task).item()
            trained_now.append(i)
            if v < exp_best[i] - 1e-6:
                exp_best[i], exp_wait[i] = v, 0
            else:
                exp_wait[i] += 1
                if exp_wait[i] >= expert_patience:
                    alive[i] = False

        # ---- rebuild the ensemble from scratch: this epoch's states + last epoch's members
        cands = [{"expert": i, "epoch": ep, "val": cur_val[i], "module": None}
                 for i in trained_now] + prev_members
        chosen, ens_val = greedy_select([c["val"] for c in cands], yva, task, max_members)
        members = []
        for j in chosen:
            c = cands[j]
            if c["module"] is None:                # snapshot only what was actually selected
                c = dict(c, module=copy.deepcopy(experts[c["expert"]]).eval())
            members.append(c)

        all_val = prob_loss(torch.stack(cur_val).mean(0), yva, task).item()
        hist["ens_val"].append(ens_val)
        hist["all_val"].append(all_val)
        hist["best_single_val"].append(min(prob_loss(c["val"], yva, task).item() for c in cands))
        hist["alive"].append(sum(alive))
        hist["ens_size"].append(len(members))

        if ens_val < best_ens - 1e-6:
            best_ens, best_epoch, ens_wait = ens_val, ep, 0
            best_members = members
            best_all = [copy.deepcopy(e).eval() for e in experts]
        else:
            ens_wait += 1
        prev_members = members
        if verbose and (ep + 1) % 25 == 0:
            say(f"    epoch {ep + 1:3d} | ens val {ens_val:.4f} | all-avg val {all_val:.4f} "
                f"| alive {sum(alive):2d}/{K} | members {len(members)}")
        if ens_wait >= ens_patience or not any(alive):
            break

    # ---- test predictions (the test set is touched only here, after selection is final)
    Xte = data["Xte"]
    ens_p = torch.stack([_predict_prob(m["module"], Xte, task, eval_batch_size)
                         for m in best_members]).mean(0)
    all_p = torch.stack([_predict_prob(m, Xte, task, eval_batch_size) for m in best_all]).mean(0)

    member_ids = [m["expert"] for m in best_members]
    return {
        "test_out": prob_to_output(ens_p, task).cpu().numpy(),
        "test_out_all": prob_to_output(all_p, task).cpu().numpy(),
        "best_epoch": best_epoch, "epochs_run": len(hist["ens_val"]),
        "seconds": time.time() - t0,
        "n_params": sum(e.num_parameters() for e in experts),
        "val_ens": best_ens, "val_all": hist["all_val"][best_epoch],
        "val_best_single": hist["best_single_val"][best_epoch],
        "ens_size": len(best_members), "n_distinct": len(set(member_ids)),
        "member_experts": member_ids, "member_epochs": [m["epoch"] for m in best_members],
        "hparams": hparams, "history": hist,
    }


# ---------------------------------------------------------------------------
# Self-test:  python hetero.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from weak_learners import WeakLearnerConfig, WeakLearnerEnsemble, default_ens

    def check(ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        return ok

    print("CHECKS")
    K, F_IN = 16, 10

    # 1. sampling
    homo = sample_hparams(K, homogeneous=True)
    check(all(h == FIXED_HPARAMS for h in homo), "homogeneous experts all get today's fixed settings")
    a, b = sample_hparams(K, seed=3), sample_hparams(K, seed=3)
    check(a == b and a != sample_hparams(K, seed=4), "sampling is deterministic per seed, differs across seeds")
    sp = HeteroSpace()
    inside = all(sp.depth[0] <= h["depth"] <= sp.depth[1]
                 and (h["dropout"] == 0.0 or sp.dropout[0] <= h["dropout"] <= sp.dropout[1])
                 and sp.lr[0] <= h["lr"] <= sp.lr[1]
                 and sp.weight_decay[0] <= h["weight_decay"] <= sp.weight_decay[1]
                 and sp.sigma[0] <= h["sigma"] <= sp.sigma[1]
                 for h in sample_hparams(500, seed=0))
    check(inside, "500 sampled experts all fall inside the declared ranges")

    # 2. same feature subsets as the control for the same seed
    ens = WeakLearnerEnsemble(F_IN, WeakLearnerConfig(**default_ens(), seed=7))
    mine = make_feature_subsets(F_IN, K, feature_frac=default_ens()["feature_frac"], seed=7)
    check(all(np.array_equal(x, y) for x, y in zip(ens.feature_subsets, mine)),
          "experts get exactly the control's feature subsets for the same seed")

    # 3. greedy selection is never worse than the best single candidate
    torch.manual_seed(0)
    yv = torch.randn(200)
    cands = [yv + torch.randn(200) * s for s in (0.3, 0.5, 0.5, 0.8, 2.0)]
    chosen, l = greedy_select(cands, yv, "regression", 5)
    single = min(prob_loss(c, yv, "regression").item() for c in cands)
    check(l <= single + 1e-9 and len(set(chosen)) == len(chosen),
          f"greedy ensemble loss {l:.4f} <= best single {single:.4f}, no repeats within a call")
    bad = [yv + 0.1, yv + 50.0]
    chosen_bad, _ = greedy_select(bad, yv, "regression", 2)
    check(chosen_bad == [0], "a uselessly bad candidate is NOT added to the ensemble")

    # 4. end-to-end on toy data, all three tasks
    torch.manual_seed(0)
    N = 600
    X = torch.randn(N, F_IN)
    signal = X[:, 0] * 2 + torch.sin(3 * X[:, 1]) - X[:, 2] * X[:, 3]
    targets = {"regression": (signal, 1),
               "binary": ((signal > 0).float(), 1),
               "multiclass": (torch.bucketize(signal, torch.tensor([-1.0, 1.0])), 3)}
    for task, (y, od) in targets.items():
        d = {"Xtr": X[:400], "ytr": y[:400], "Xva": X[400:500], "yva": y[400:500], "Xte": X[500:]}
        r = train_hetero(d, task=task, output_dim=od, n_features=F_IN,
                         hparams=sample_hparams(6, seed=1), seed=0, epochs=15, batch_size=64,
                         ens_patience=5, expert_patience=3)
        hist = r["history"]
        ok_sel = all(e <= s + 1e-6 for e, s in zip(hist["ens_val"], hist["best_single_val"]))
        shape_ok = r["test_out"].shape == ((100,) if od == 1 else (100, od))
        check(ok_sel and shape_ok and np.isfinite(r["test_out"]).all(),
              f"{task:10s}: trains, ensemble val <= best single every epoch, test output "
              f"{r['test_out'].shape}, {r['ens_size']} members ({r['n_distinct']} distinct experts)")

    # 5. a retired expert is frozen
    d = {"Xtr": X[:400], "ytr": signal[:400], "Xva": X[400:500], "yva": signal[400:500], "Xte": X[500:]}
    torch.manual_seed(0)
    e = HeteroExpert(np.arange(4), 1, 1, 0.0, 0.1)
    opt = torch.optim.AdamW([{"params": e.parameters(), "lr": 1e-2, "weight_decay": 0.1}])
    before = [p.detach().clone() for p in e.parameters()]
    opt.zero_grad(set_to_none=True)      # no loss term for this expert -> grads stay None
    opt.step()
    check(all(torch.equal(p, q) for p, q in zip(before, e.parameters())),
          "an expert left out of the loss is not touched by AdamW (not even weight decay)")

    # 6. at fixed settings an expert is one v4 expert + its own embedding and head
    e = HeteroExpert(np.arange(6), 1, **{k: FIXED_HPARAMS[k] for k in ("depth", "dropout", "sigma")})
    body_ref = nn.Sequential(nn.Linear(6 * 8, 16), nn.GELU(), nn.Dropout(0.1), nn.Linear(16, 32))
    n_body = sum(p.numel() for p in e.body.parameters())
    check(n_body == sum(p.numel() for p in body_ref.parameters()),
          f"fixed-settings expert body matches WeakLearnerMLP's shape ({n_body:,} params)")
