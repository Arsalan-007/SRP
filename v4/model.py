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
    conf_uniform    predict-then-average, ensemble-level loss                   (confidence control)
    conf_static     + learned per-expert weight, same for every row             (is any expert better?)
    conf_dynamic    + per-row confidence from each expert's own token           (does weighting need x?)
    conf_sigmoid    + per-row INDEPENDENT confidence, explicit / sum(c)      (non-competitive)
    conf_gumbel     + conf_dynamic with Gumbel noise while training          (does exploring help?)

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
from ncl import ncl_loss

__all__ = [
    "SRPConfig",
    "SRPModel",
    "PlainMLP",
    "PredictionHead",
    "step_weights",
    "make_arm",
    "ARM_NAMES",
    "CONFIDENCE_MODES",
]

AGGREGATORS = ("meanpool", "causal", "bidirectional")
READOUTS = ("last", "mean")
TASKS = ("regression", "binary", "multiclass")
STEP_WEIGHT_SCHEMES = ("uniform", "linear", "exponential")
POOL_LEVELS = ("embedding", "output")
CONFIDENCE_MODES = ("none", "uniform", "static", "dynamic", "sigmoid", "gumbel")
ARM_NAMES = ("meanpool", "meanpool_wide", "meanpool_ncl", "conf_uniform", "conf_static",
             "conf_dynamic", "conf_sigmoid", "conf_gumbel",
             "bidirectional", "causal", "causal_ds")


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
    pool_level          : "embedding" (default) pools expert EMBEDDINGS, then runs the head
                         ONCE — today's v4, unchanged. "output" runs the (shared) head on
                         EVERY expert's embedding first, then averages the K scalar
                         predictions — required for NCL (see ncl.py), since NCL needs each
                         expert's own prediction, not just its own opinion vector. Only
                         wired up for aggregator="meanpool" so far.
    ncl_lambda          : Negative Correlation Learning penalty strength (ncl.py). Only
                         used when pool_level="output". 0 = plain independent per-learner
                         training under output-level pooling (isolates the pooling-point
                         change from the penalty itself).
    confidence          : per-expert aggregation weights, replacing the plain mean over the K
                         per-expert predictions. Needs pool_level="output" (each expert must
                         have its own scalar prediction to weight).
                           "none"    today's v4, unchanged — plain mean, per-expert NCL loss.
                           "uniform" fixed 1/K weights. Identical forward to "none"; the ONLY
                                     difference is that the loss is the criterion on the
                                     ENSEMBLE prediction, not the mean of per-expert losses.
                                     This is the control the learned modes are compared against
                                     — without it, a win could be the objective, not the weights.
                           "static"  one learned logit per expert (K params), same for every row.
                           "dynamic" a logit per expert per row, from that expert's own token
                                     (embed_dim+1 params). The supervisor's c_i.
                           "sigmoid" a per-row confidence per expert, each judged INDEPENDENTLY
                                     (sigmoid, not softmax), then divided by their sum. This is
                                     the supervisor's `sum(y_i c_i) / sum(c_i)` in its literal
                                     form, and it differs from "dynamic" in kind, not degree:
                                     softmax is competitive (raising one expert's weight must
                                     lower another's) and cannot express "every expert is unsure
                                     about this row"; independent confidences can.
                           "gumbel"  "dynamic" + Gumbel noise during training only. Soft
                                     (hard=False), so it explores the same weight space rather
                                     than hard-routing each row to a single expert.
                         Weights are softmax-normalised, so they sum to 1 and the supervisor's
                         `/ sum(c)` denominator is already accounted for.
    conf_tau            : Gumbel-Softmax temperature. Lower = sparser weights.
    ncl_detach_mean     : see ncl.py's module docstring — provably makes no difference under
                         this codebase's single-joint-backward training loop; kept for
                         completeness / anyone training learners with separate updates.
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
    pool_level: str = "embedding"
    ncl_lambda: float = 0.0
    ncl_detach_mean: bool = False
    confidence: str = "none"
    conf_tau: float = 1.0

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
        if self.confidence not in CONFIDENCE_MODES:
            raise ValueError(f"confidence must be one of {CONFIDENCE_MODES}, got {self.confidence!r}")
        if self.confidence != "none" and self.pool_level != "output":
            raise ValueError("confidence weighting needs pool_level='output' — each expert must "
                             "have its own scalar prediction before it can be weighted")
        if self.confidence != "none" and self.ncl_lambda != 0.0:
            # Not a limitation of the maths, a deliberate scope cut: NCL's penalty pulls experts
            # away from the UNWEIGHTED mean, so combining the two makes "which effect won?"
            # unanswerable. Run them as separate arms first.
            raise ValueError("confidence weighting and ncl_lambda>0 are not combinable yet; "
                             "run them as separate arms")
        if self.pool_level not in POOL_LEVELS:
            raise ValueError(f"pool_level must be one of {POOL_LEVELS}, got {self.pool_level!r}")
        if self.pool_level == "output" and self.aggregator != "meanpool":
            raise ValueError("pool_level='output' is only wired up for aggregator='meanpool' so far")
        if self.pool_level == "output" and self.task == "multiclass":
            raise ValueError("ncl.py's penalty is only defined for scalar predictions "
                             "(regression/binary) — multiclass pool_level='output' is not yet supported")
        if self.pool_level == "embedding" and (self.ncl_lambda != 0.0 or self.ncl_detach_mean):
            raise ValueError("ncl_lambda/ncl_detach_mean only apply when pool_level='output'")


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
                 config: Optional[SRPConfig] = None,
                 bin_edges=None):
        super().__init__()
        self.config = config or SRPConfig()
        c = self.config
        D, K = ensemble_config.embed_dim, ensemble_config.num_learners

        # Constructed FIRST so a fixed torch seed gives identical expert weights in every arm.
        # The embedding's width is read from the config (not from the built module) so that
        # building the embedding cannot shift the RNG before the experts are initialised.
        self.ensemble = WeakLearnerEnsemble(in_dim, ensemble_config,
                                            in_per_feature=c.embedding.out_per_feature)
        # bin_edges: only needed for embedding.mode="piecewise_linear" — fixed, fit-from-data
        # bin boundaries (embeddings.py's compute_quantile_bins), not a learned parameter.
        self.embedding = FeatureEmbedding(in_dim, c.embedding, bin_edges=bin_edges)

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

        # "uniform" is deliberately parameter-free: it differs from "none" only in the LOSS
        # (see .loss()), which is what makes it a clean control for the learned modes.
        if c.confidence in ("static",):
            self.conf = nn.Parameter(torch.zeros(K))        # softmax(0) = exactly 1/K at init
        elif c.confidence in ("dynamic", "sigmoid", "gumbel"):
            self.conf = nn.Linear(D, 1)
            nn.init.zeros_(self.conf.weight); nn.init.zeros_(self.conf.bias)
        else:
            self.conf = None

    # -- forward --------------------------------------------------------------
    def _squeeze(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(-1) if self.config.output_dim == 1 else y

    def _conf_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-expert aggregation weights, (B, K), each row summing to 1."""
        c = self.config
        B = tokens.shape[0]
        if c.confidence == "static":
            logits = self.conf.unsqueeze(0).expand(B, -1)
        else:
            logits = self.conf(tokens).squeeze(-1)
        # Gumbel noise is a TRAINING-time exploration device. Sampling it at eval would make
        # test predictions differ run to run for the same weights.
        if c.confidence == "sigmoid":
            # The supervisor's formula literally: independent c_i in (0,1), then / sum(c).
            # sigmoid(0)=0.5 for every expert, so zero-init still starts at exactly 1/K.
            conf = torch.sigmoid(logits)
            return conf / conf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if c.confidence == "gumbel" and self.training:
            return F.gumbel_softmax(logits, tau=c.conf_tau, hard=False, dim=-1)
        return logits.softmax(dim=-1)

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Dict[str, torch.Tensor]:
        c = self.config
        tokens = self.ensemble(self.embedding(x), x_raw=x)
        out = {"tokens": tokens, "context": None, "attn": None, "step_preds": None, "conf": None}

        if self.stage is None:
            if c.pool_level == "output":
                steps = self._squeeze(self.head(tokens))    # (B, K) — head applied to EVERY expert
                out["step_preds"] = steps
                if c.confidence in ("none", "uniform"):
                    out["pred"] = steps.mean(dim=1)
                else:
                    w = self._conf_weights(tokens)
                    out["conf"] = w
                    out["pred"] = (steps * w).sum(dim=1)
            else:
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
        """Training objective: criterion on `pred`, the NCL loss under output-level
        mean-pooling, or the weighted sum over positions under deep supervision."""
        steps = out["step_preds"]
        if steps is None:
            return self.criterion(out["pred"], y)
        if self.config.confidence != "none":
            # MUST be the ensemble prediction: ncl_loss below only touches step_preds, so
            # under it the aggregation weights would receive zero gradient and sit at their
            # uniform init forever — the arm would silently be a copy of meanpool_ncl.
            return self.criterion(out["pred"], y)
        if self.config.aggregator == "meanpool" and self.config.pool_level == "output":
            return ncl_loss(steps, y, self.criterion, lambda_ncl=self.config.ncl_lambda,
                            detach_mean=self.config.ncl_detach_mean)
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
             embedding: Optional[EmbeddingConfig] = None,
             ncl_lambda: float = 0.0,
             ncl_detach_mean: bool = False,
             conf_tau: float = 1.0,
             bin_edges=None) -> SRPModel:
    """Build one arm of the ablation. Set torch.manual_seed(seed) right before calling
    this to get identical expert weights across arms for that seed.

    bin_edges: required iff embedding.mode="piecewise_linear" — see embeddings.py's
    compute_quantile_bins(). Ignored otherwise."""
    common = dict(task=task, output_dim=output_dim, embedding=embedding or EmbeddingConfig())

    if name == "meanpool":
        cfg = SRPConfig(aggregator="meanpool", **common)
    elif name == "meanpool_ncl":
        # ncl_lambda=0 isolates the pooling-point change (predict-then-average) from the
        # penalty itself — compare against "meanpool" to see if THAT alone matters.
        cfg = SRPConfig(aggregator="meanpool", pool_level="output",
                        ncl_lambda=ncl_lambda, ncl_detach_mean=ncl_detach_mean, **common)
    elif name in ("conf_uniform", "conf_static", "conf_dynamic", "conf_sigmoid", "conf_gumbel"):
        # Same pooling point as meanpool_ncl; only the weighting and the loss differ.
        # conf_uniform is the control — compare the others against IT, not against meanpool.
        mode = {"conf_uniform": "uniform", "conf_static": "static", "conf_dynamic": "dynamic",
                "conf_sigmoid": "sigmoid", "conf_gumbel": "gumbel"}[name]
        cfg = SRPConfig(aggregator="meanpool", pool_level="output", confidence=mode,
                        conf_tau=conf_tau, **common)
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
                           SRPConfig(aggregator="causal", **common), bin_edges=bin_edges)
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

    return SRPModel(in_dim, ensemble_config, attention_config, cfg, bin_edges=bin_edges)


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

    # 9. meanpool_ncl at lambda=0: same architecture as meanpool (identical param count),
    #    predict-then-average pooling instead of average-then-predict, step_preds present.
    ok9a = arms["meanpool_ncl"].num_parameters() == arms["meanpool"].num_parameters()
    m = arms["meanpool_ncl"].eval()
    with torch.no_grad():
        o = m(x)
    ok9b = o["step_preds"] is not None and tuple(o["step_preds"].shape) == (B, ENS.num_learners)
    ok9c = torch.allclose(o["pred"], o["step_preds"].mean(dim=1), atol=1e-6)
    print(f"  [{'PASS' if ok9a else 'FAIL'}] meanpool_ncl has the SAME param count as meanpool "
          f"({arms['meanpool_ncl'].num_parameters():,} vs {arms['meanpool'].num_parameters():,}) "
          f"— pooling point is a loss/forward change only, not a capacity change")
    print(f"  [{'PASS' if ok9b and ok9c else 'FAIL'}] meanpool_ncl produces per-learner step_preds "
          f"{tuple(o['step_preds'].shape)} and pred == their mean")

    # 10. meanpool_ncl(lambda=0) loss == mean of independent per-learner criterion calls
    #     (bit-identical to plain independent training — the pooling-point isolation check).
    m = arms["meanpool_ncl"]
    out = m(x)
    got = m.loss(out, y)
    want = torch.stack([m.criterion(out["step_preds"][:, i], y) for i in range(ENS.num_learners)]).mean()
    ok10 = torch.allclose(got, want, atol=1e-6)
    print(f"  [{'PASS' if ok10 else 'FAIL'}] meanpool_ncl(lambda=0) loss exactly equals independent "
          f"per-learner training ({got.item():.4f} vs {want.item():.4f})")

    # 10b. confidence weighting: params, normalisation, control-arm equivalence, gradients
    torch.manual_seed(0); cu = make_arm("conf_uniform", IN_DIM, ENS, ATT)
    torch.manual_seed(0); cs = make_arm("conf_static", IN_DIM, ENS, ATT)
    torch.manual_seed(0); cd = make_arm("conf_dynamic", IN_DIM, ENS, ATT)
    torch.manual_seed(0); ncl_ref = make_arm("meanpool_ncl", IN_DIM, ENS, ATT)
    D_ = ENS.embed_dim
    okA = (cu.num_parameters() == ncl_ref.num_parameters()
           and cs.num_parameters() == cu.num_parameters() + ENS.num_learners
           and cd.num_parameters() == cu.num_parameters() + D_ + 1)
    print(f"  [{'PASS' if okA else 'FAIL'}] conf arms cost what they should "
          f"(uniform +0, static +K={ENS.num_learners}, dynamic +D+1={D_ + 1})")

    for m in (cu, cs, cd, ncl_ref):
        m.eval()
    with torch.no_grad():
        ou, os_, od, oref = cu(x), cs(x), cd(x), ncl_ref(x)
    # uniform must be a pure loss-level change: its forward is bit-identical to meanpool_ncl's.
    okB = torch.equal(ou["pred"], oref["pred"]) and ou["conf"] is None
    print(f"  [{'PASS' if okB else 'FAIL'}] conf_uniform forward == meanpool_ncl forward "
          f"(it is a LOSS-level control, not an architecture change)")
    okC = all(torch.allclose(o["conf"].sum(dim=1), torch.ones(B), atol=1e-5) for o in (os_, od))
    print(f"  [{'PASS' if okC else 'FAIL'}] learned conf weights are softmax-normalised "
          f"(rows sum to 1, so the supervisor's /sum(c) is already applied)")
    # Zero-init => both learned modes START at exactly uniform, so they can only improve on it.
    okD = torch.allclose(os_["pred"], oref["pred"], atol=1e-6) and torch.allclose(od["pred"], oref["pred"], atol=1e-6)
    print(f"  [{'PASS' if okD else 'FAIL'}] zero-init puts conf_static/conf_dynamic exactly at "
          f"mean-pooling before training")

    # The whole point of the loss override: without it these grads are None/zero.
    grads = {}
    for nm, m in (("static", cs), ("dynamic", cd)):
        m.train(); m.zero_grad()
        m.loss(m(x), y).backward()
        g = m.conf.grad if nm == "static" else m.conf.weight.grad
        grads[nm] = g is not None and g.abs().sum().item() > 0
    okE = all(grads.values())
    print(f"  [{'PASS' if okE else 'FAIL'}] confidence weights actually RECEIVE gradient "
          f"(static={grads['static']}, dynamic={grads['dynamic']}) — ncl_loss alone would not train them")

    # gumbel must be deterministic at eval, or test numbers move run to run
    torch.manual_seed(0); cg = make_arm("conf_gumbel", IN_DIM, ENS, ATT)
    cg.eval()
    with torch.no_grad():
        okF = torch.equal(cg(x)["pred"], cg(x)["pred"])
    print(f"  [{'PASS' if okF else 'FAIL'}] conf_gumbel is deterministic in eval mode "
          f"(noise is training-only)")
    cg.train()
    with torch.no_grad():
        okF2 = not torch.equal(cg(x)["pred"], cg(x)["pred"])
    print(f"  [{'PASS' if okF2 else 'FAIL'}] conf_gumbel DOES sample while training "
          f"(otherwise it would silently be conf_dynamic)")

    # 10c. sigmoid mode: the supervisor's literal sum(y*c)/sum(c), independent confidences
    torch.manual_seed(0); csg = make_arm("conf_sigmoid", IN_DIM, ENS, ATT)
    okG = csg.num_parameters() == cd.num_parameters()      # same Linear(D,1) cost as dynamic
    csg.eval()
    with torch.no_grad():
        osg = csg(x)
    okH = torch.allclose(osg["conf"].sum(dim=1), torch.ones(B), atol=1e-5)
    okI = torch.allclose(osg["pred"], oref["pred"], atol=1e-6)   # sigmoid(0)=0.5 -> uniform
    print(f"  [{'PASS' if okG and okH and okI else 'FAIL'}] conf_sigmoid: same cost as dynamic, "
          f"weights sum to 1 after the explicit /sum(c), and zero-init starts at mean-pooling")
    # Independence is the WHOLE point vs softmax: raising one logit must NOT force others down
    # before normalisation. Check the pre-normalisation confidences are not coupled.
    with torch.no_grad():
        lg = csg.conf(csg.ensemble(csg.embedding(x))).squeeze(-1)
        raw = torch.sigmoid(lg)
        bumped = torch.sigmoid(lg.clone().index_fill_(1, torch.tensor([0]), 5.0))
    okJ = torch.allclose(raw[:, 1:], bumped[:, 1:], atol=1e-6)
    print(f"  [{'PASS' if okJ else 'FAIL'}] conf_sigmoid confidences are INDEPENDENT "
          f"(raising expert 0's logit leaves the others' raw confidence unchanged, unlike softmax)")

    for nm, m in (("sigmoid", csg), ("gumbel", cg)):
        m.train(); m.zero_grad()
        m.loss(m(x), y).backward()
        g = m.conf.weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"{nm} conf got no gradient"
    print(f"  [PASS] conf_sigmoid and conf_gumbel weights both receive gradient")

    # 11. invalid combinations are rejected loudly, not silently ignored
    def _raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True
    ok11a = _raises(lambda: SRPConfig(aggregator="causal", pool_level="output"))
    ok11b = _raises(lambda: SRPConfig(aggregator="meanpool", pool_level="output",
                                      task="multiclass", output_dim=5))
    ok11c = _raises(lambda: SRPConfig(aggregator="meanpool", pool_level="embedding", ncl_lambda=0.5))
    ok11d = _raises(lambda: SRPConfig(aggregator="meanpool", confidence="static"))
    ok11e = _raises(lambda: SRPConfig(aggregator="meanpool", pool_level="output",
                                      confidence="static", ncl_lambda=0.5))
    print(f"  [{'PASS' if ok11a and ok11b and ok11c and ok11d and ok11e else 'FAIL'}] invalid "
          f"pool_level/task/ncl_lambda/confidence combinations raise ValueError instead of "
          f"silently doing the wrong thing")
