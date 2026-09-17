"""
model.py  —  SRP v4, step 3

The full model:

    x (B, F) --ensemble--> tokens (B, K, D) --aggregator--> z --head--> prediction

Only the AGGREGATOR differs between the experimental arms. The weak learners, the
prediction head, and the training contract are shared, so any difference in results
is attributable to how the K expert opinions are combined.

    meanpool        tokens.mean(K)                    no attention, no order   (step-1 control)
    meanpool_wide   same, head widened to attention's parameter count          (capacity control)
    bidirectional   attention, causal=False, read last token                   (attention without order)
    causal          attention, causal=True,  read last token                   (the staircase)
    causal_ds       causal + exponentially-weighted deep supervision           (v2's full recipe)

Build arms with `make_arm()`. For a fixed torch seed, every arm gets bit-identical
weak-learner weights (the ensemble is constructed first, and capacity matching runs
under a forked RNG), which is what makes per-seed paired comparisons valid.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from weak_learners import WeakLearnerConfig, WeakLearnerEnsemble
from attention import AttentionConfig, CausalAttentionStage
from embeddings import EmbeddingConfig, FeatureEmbedding

__all__ = [
    "SRPConfig",
    "SRPModel",
    "PlainMLP",
    "PredictionHead",
    "step_weights",
    "make_arm",
    "ARM_NAMES",
]

AGGREGATORS = ("meanpool", "causal", "bidirectional")
READOUTS = ("last", "mean")
TASKS = ("regression", "binary", "multiclass")
STEP_WEIGHT_SCHEMES = ("uniform", "linear", "exponential")
ARM_NAMES = ("meanpool", "meanpool_wide", "bidirectional", "causal", "causal_ds")


# ---------------------------------------------------------------------------
# Deep-supervision weights
# ---------------------------------------------------------------------------
def step_weights(num_steps: int, scheme: str = "exponential",
                 base: float = 2.0, device=None) -> torch.Tensor:
    """Normalised per-position loss weights, later positions weighted more (except uniform).

    exponential with base 2 is v2's scheme: w_i ∝ 2^i. With K=16 the last position
    carries ~50% of the loss, the one before ~25%, and position 0 ~0.002%.
    """
    if scheme == "uniform":
        w = torch.ones(num_steps)
    elif scheme == "linear":
        w = torch.arange(1, num_steps + 1, dtype=torch.float32)
    elif scheme == "exponential":
        w = torch.pow(torch.tensor(float(base)), torch.arange(num_steps, dtype=torch.float32))
    else:
        raise ValueError(f"unknown step weight scheme {scheme!r}; choose from {STEP_WEIGHT_SCHEMES}")
    w = w.to(device=device, dtype=torch.float32)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------
class PredictionHead(nn.Module):
    """LayerNorm -> [Linear -> SiLU -> Dropout] x depth -> Linear(output_dim).

    With depth=1 and hidden=embed_dim this is exactly step 1's head, so the
    `meanpool` arm reproduces the step-1 WeakEnsemble-MeanPool model.
    """

    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, depth: int = 1,
                 output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        layers = [nn.LayerNorm(in_dim)]
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _head_param_count(in_dim: int, hidden: int, depth: int, output_dim: int) -> int:
    """Closed-form parameter count of PredictionHead (verified in the smoke test)."""
    n = 2 * in_dim                                    # LayerNorm
    if depth == 0:
        return n + in_dim * output_dim + output_dim
    n += in_dim * hidden + hidden                     # first Linear
    n += (depth - 1) * (hidden * hidden + hidden)     # further hidden Linears
    n += hidden * output_dim + output_dim             # output Linear
    return n


def head_width_for_budget(in_dim: int, budget: int, depth: int = 2,
                          output_dim: int = 1, max_width: int = 8192) -> int:
    """Hidden width whose head parameter count is closest to `budget`."""
    best_w, best_err = 1, float("inf")
    for w in range(1, max_width + 1):
        err = abs(_head_param_count(in_dim, w, depth, output_dim) - budget)
        if err < best_err:
            best_w, best_err = w, err
        elif _head_param_count(in_dim, w, depth, output_dim) > budget:
            break                                     # monotonic in w: past the target
    return best_w


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SRPConfig:
    """How the ensemble's tokens become a prediction.

    aggregator         : "meanpool" | "causal" | "bidirectional"
    readout            : "last" (token K-1) | "mean" (average of contextualised tokens).
                         Ignored for meanpool.
    head_hidden        : head width; None -> embed_dim.
    head_depth         : hidden layers in the head.
    head_dropout       : dropout inside the head.
    deep_supervision   : every position predicts and is trained (attention only).
                         Reported prediction is always the LAST position — the same
                         quantity used for early stopping and for the final table.
    step_weight_scheme : per-position loss weights under deep supervision.
    task / output_dim  : regression & binary -> 1; multiclass -> n_classes.
    """

    aggregator: str = "causal"
    readout: str = "last"
    head_hidden: Optional[int] = None
    head_depth: int = 1
    head_dropout: float = 0.1
    deep_supervision: bool = False
    step_weight_scheme: str = "exponential"
    task: str = "regression"
    output_dim: int = 1
    # Per-feature embeddings (see embeddings.py). Default "none" = the original v4 model,
    # bit-for-bit: the module is the identity and has no parameters.
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def __post_init__(self):
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}, got {self.aggregator!r}")
        if self.readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {self.readout!r}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.step_weight_scheme not in STEP_WEIGHT_SCHEMES:
            raise ValueError(f"unknown step_weight_scheme {self.step_weight_scheme!r}")
        if self.deep_supervision and self.aggregator == "meanpool":
            raise ValueError("deep_supervision needs a token sequence; meanpool has none")
        if self.deep_supervision and self.readout != "last":
            raise ValueError("deep_supervision reports the last position; use readout='last'")
        if self.task in ("regression", "binary") and self.output_dim != 1:
            raise ValueError(f"{self.task} needs output_dim=1, got {self.output_dim}")
        if self.task == "multiclass" and self.output_dim < 2:
            raise ValueError("multiclass needs output_dim >= 2")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SRPModel(nn.Module):
    """Weak learners -> aggregator -> head.

    forward(x) returns a dict so every arm exposes the same interface:
        pred        (B,) or (B, C)       final prediction (always what gets reported)
        tokens      (B, K, D)            raw expert embeddings (for diversity diagnostics)
        context     (B, K, D) | None     contextualised tokens (attention arms)
        attn        list[(B, K, K)] | None   per-block attention (only if need_weights)
        step_preds  (B, K) | (B, K, C) | None   per-position predictions (deep supervision)
    """

    def __init__(self, in_dim: int, ensemble_config: WeakLearnerConfig,
                 attention_config: Optional[AttentionConfig] = None,
                 config: Optional[SRPConfig] = None):
        super().__init__()
        self.config = config or SRPConfig()
        c = self.config
        D, K = ensemble_config.embed_dim, ensemble_config.num_learners

        # Constructed FIRST so a fixed torch seed gives identical expert weights in every arm.
        # The embedding's width is read from the config (not from the built module) so that
        # building the embedding cannot shift the RNG before the experts are initialised.
        self.ensemble = WeakLearnerEnsemble(in_dim, ensemble_config,
                                            in_per_feature=c.embedding.out_per_feature)
        self.embedding = FeatureEmbedding(in_dim, c.embedding)

        if c.aggregator == "meanpool":
            self.stage = None
        else:
            base = attention_config or AttentionConfig(embed_dim=D)
            # embed_dim must match the experts; causality is decided by the aggregator, not
            # by whatever the attention config happened to say.
            acfg = replace(base, embed_dim=D, causal=(c.aggregator == "causal"))
            # max_len = K: no unused step-embedding rows inflating the parameter count.
            self.stage = CausalAttentionStage(acfg, max_len=K)

        self.head = PredictionHead(D, c.head_hidden, c.head_depth, c.output_dim, c.head_dropout)

    # -- forward --------------------------------------------------------------
    def _squeeze(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(-1) if self.config.output_dim == 1 else y

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Dict[str, torch.Tensor]:
        c = self.config
        tokens = self.ensemble(self.embedding(x))
        out = {"tokens": tokens, "context": None, "attn": None, "step_preds": None}

        if self.stage is None:
            out["pred"] = self._squeeze(self.head(tokens.mean(dim=1)))
            return out

        ctx, attn = self.stage(tokens, need_weights=need_weights)
        out["context"], out["attn"] = ctx, attn

        if c.deep_supervision:
            steps = self._squeeze(self.head(ctx))          # (B, K) or (B, K, C)
            out["step_preds"] = steps
            out["pred"] = steps[:, -1]
        else:
            z = ctx[:, -1] if c.readout == "last" else ctx.mean(dim=1)
            out["pred"] = self._squeeze(self.head(z))
        return out

    # -- losses ---------------------------------------------------------------
    def criterion(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss for ONE prediction. Used for early stopping and evaluation in every arm."""
        t = self.config.task
        if t == "regression":
            return F.mse_loss(pred.float(), y.float())
        if t == "binary":
            return F.binary_cross_entropy_with_logits(pred.float(), y.float())
        return F.cross_entropy(pred, y.long())

    def loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """Training objective: criterion on `pred`, or the weighted sum over positions
        under deep supervision."""
        steps = out["step_preds"]
        if steps is None:
            return self.criterion(out["pred"], y)
        K = steps.shape[1]
        w = step_weights(K, self.config.step_weight_scheme, device=steps.device)
        return sum(w[i] * self.criterion(steps[:, i], y) for i in range(K))

    # -- introspection --------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        c = self.config
        return (f"aggregator={c.aggregator}, readout={c.readout}, "
                f"deep_supervision={c.deep_supervision}, head={c.head_depth}x{c.head_hidden}, "
                f"task={c.task}, embedding={c.embedding.mode}")


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------
class PlainMLP(nn.Module):
    """One conventional MLP — the neural reference line in every benchmark.

    Returns a raw prediction tensor (not a dict) and exposes `criterion`, so the shared
    training loop treats it exactly like an SRPModel. Same architecture as the inline
    PlainMLP of notebooks 01-02, with classification support added.
    """

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3, dropout: float = 0.1,
                 task: str = "regression", output_dim: int = 1):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        self.task, self.output_dim = task, output_dim
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.squeeze(-1) if self.output_dim == 1 else y

    def criterion(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":
            return F.mse_loss(pred.float(), y.float())
        if self.task == "binary":
            return F.binary_cross_entropy_with_logits(pred.float(), y.float())
        return F.cross_entropy(pred, y.long())


# ---------------------------------------------------------------------------
# Experimental arms
# ---------------------------------------------------------------------------
def make_arm(name: str, in_dim: int, ensemble_config: WeakLearnerConfig,
             attention_config: Optional[AttentionConfig] = None,
             task: str = "regression", output_dim: int = 1,
             step_weight_scheme: str = "exponential",
             wide_head_depth: int = 2,
             embedding: Optional[EmbeddingConfig] = None) -> SRPModel:
    """Build one arm of the ablation. Set torch.manual_seed(seed) right before calling
    this to get identical expert weights across arms for that seed."""
    common = dict(task=task, output_dim=output_dim, embedding=embedding or EmbeddingConfig())

    if name == "meanpool":
        cfg = SRPConfig(aggregator="meanpool", **common)
    elif name == "bidirectional":
        cfg = SRPConfig(aggregator="bidirectional", readout="last", **common)
    elif name == "causal":
        cfg = SRPConfig(aggregator="causal", readout="last", **common)
    elif name == "causal_ds":
        cfg = SRPConfig(aggregator="causal", readout="last", deep_supervision=True,
                        step_weight_scheme=step_weight_scheme, **common)
    elif name == "meanpool_wide":
        # Size the head so the whole model matches an attention arm's parameter count.
        # Runs under a forked RNG so building the reference model doesn't shift the
        # global torch seed (which would break identical expert init across arms).
        with torch.random.fork_rng(devices=[]):
            ref = SRPModel(in_dim, ensemble_config, attention_config,
                           SRPConfig(aggregator="causal", **common))
            # The experts and the embedding are identical in both arms, so this head's budget
            # is everything else the causal arm spends: its attention stage plus its own head.
            budget = (ref.num_parameters() - ref.ensemble.num_parameters()
                      - ref.embedding.num_parameters())
            del ref
        width = head_width_for_budget(ensemble_config.embed_dim, budget,
                                      depth=wide_head_depth, output_dim=output_dim)
        cfg = SRPConfig(aggregator="meanpool", head_hidden=width,
                        head_depth=wide_head_depth, **common)
    else:
        raise ValueError(f"unknown arm {name!r}; choose from {ARM_NAMES}")

    return SRPModel(in_dim, ensemble_config, attention_config, cfg)


# ---------------------------------------------------------------------------
# Smoke test:  python model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    IN_DIM, B = 8, 32
    ENS = WeakLearnerConfig(num_learners=16, hidden_dim=16, embed_dim=32,
                            depth=1, dropout=0.1, feature_frac=0.7, seed=0)
    ATT = AttentionConfig(embed_dim=32, num_heads=4, attn_depth=2, need_weights=False)
    x, y = torch.randn(B, IN_DIM), torch.randn(B)

    print("ARMS  (same seed -> same experts)")
    print("-" * 72)
    arms, ens_states = {}, {}
    for name in ARM_NAMES:
        torch.manual_seed(0)
        m = make_arm(name, IN_DIM, ENS, ATT)
        arms[name] = m
        ens_states[name] = {k: v.clone() for k, v in m.ensemble.state_dict().items()}
        out = m(x)
        loss = m.loss(out, y)
        sp = out["step_preds"]
        print(f"  {name:14s} params={m.num_parameters():7,d}   pred {tuple(out['pred'].shape)}"
              f"   step_preds {str(tuple(sp.shape)) if sp is not None else '—':>9}   loss {loss.item():.3f}")

    # 1. meanpool reproduces step 1's model exactly
    print("\nCHECKS")
    ok1 = arms["meanpool"].num_parameters() == 11_649
    print(f"  [{'PASS' if ok1 else 'FAIL'}] meanpool == step-1 WeakEnsemble-MeanPool (11,649 params)")

    # 2. capacity matching
    att_n = arms["causal"].num_parameters()
    wide_n = arms["meanpool_wide"].num_parameters()
    gap = abs(wide_n - att_n) / att_n
    print(f"  [{'PASS' if gap < 0.01 else 'FAIL'}] meanpool_wide within 1% of attention arms "
          f"({wide_n:,} vs {att_n:,}, {gap:.2%})")
    same_size = arms["causal"].num_parameters() == arms["bidirectional"].num_parameters() \
        == arms["causal_ds"].num_parameters()
    print(f"  [{'PASS' if same_size else 'FAIL'}] causal, bidirectional, causal_ds identical size")

    # 3. identical expert init across arms
    ref = ens_states["meanpool"]
    same_init = all(all(torch.equal(ref[k], s[k]) for k in ref) for s in ens_states.values())
    print(f"  [{'PASS' if same_init else 'FAIL'}] all arms share bit-identical expert weights for a seed")

    # 4. closed-form head count matches reality
    h = PredictionHead(32, 148, 2, 1, 0.1)
    ok4 = sum(p.numel() for p in h.parameters()) == _head_param_count(32, 148, 2, 1)
    print(f"  [{'PASS' if ok4 else 'FAIL'}] closed-form head parameter count is exact")

    # 5. causal flag actually wired through
    ok5 = arms["causal"].stage.config.causal and not arms["bidirectional"].stage.config.causal
    print(f"  [{'PASS' if ok5 else 'FAIL'}] aggregator controls causality (causal=True / bidirectional=False)")

    # 6. need_weights on/off gives the same outputs (fast path vs slow path)
    m = arms["causal"].eval()
    with torch.no_grad():
        a, b = m(x, need_weights=False)["pred"], m(x, need_weights=True)["pred"]
    ok6 = torch.allclose(a, b, atol=1e-5)
    print(f"  [{'PASS' if ok6 else 'FAIL'}] need_weights=False/True give identical predictions")

    # 7. deep supervision reports the last position
    m = arms["causal_ds"].eval()
    with torch.no_grad():
        o = m(x)
    ok7 = torch.equal(o["pred"], o["step_preds"][:, -1])
    print(f"  [{'PASS' if ok7 else 'FAIL'}] causal_ds pred == last step prediction")

    # 8. other tasks
    torch.manual_seed(0)
    mb = make_arm("causal_ds", IN_DIM, ENS, ATT, task="binary")
    lb = mb.loss(mb(x), (torch.rand(B) > 0.5).float())
    mc = make_arm("causal_ds", IN_DIM, ENS, ATT, task="multiclass", output_dim=5)
    oc = mc(x)
    lc = mc.loss(oc, torch.randint(0, 5, (B,)))
    ok8 = torch.isfinite(lb) and torch.isfinite(lc) and oc["step_preds"].shape == (B, 16, 5)
    print(f"  [{'PASS' if ok8 else 'FAIL'}] binary and multiclass losses finite; "
          f"multiclass step_preds {tuple(oc['step_preds'].shape)}")

    print(f"\n  exponential step weights (K=16): first={step_weights(16)[0]:.2e}  "
          f"last={step_weights(16)[-1]:.3f}  last-3 share={step_weights(16)[-3:].sum():.3f}")
