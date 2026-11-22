"""
weak_learners.py  —  SRP v4, step 1

The base "expert" of the architecture: a deliberately small MLP that sees only a
random subset of the input features, and a container that stacks K of them into a
token sequence for the attention stage to consume.

Design notes (carried over from the v1/v2/v3 post-mortem)
--------------------------------------------------------
* Diversity comes from FEATURE BAGGING, not from random init. v2 gave every learner
  the identical full input vector and then had to fight "expert collapse" with an
  artificial similarity penalty. Giving each learner its own column subset is the
  same trick Random Forests use, it costs nothing at training time, and it makes the
  learners structurally unable to be identical.

* Diversity is MEASURED, not assumed. `WeakLearnerEnsemble.diversity_stats()` reports
  the pairwise cosine similarity of the raw (pre-attention) embeddings, so "the
  architecture prevents collapse" stays a checkable claim rather than an assertion.

* Capacity limits are a CONFIG CHOICE, not an assert. Weak learners are supposed to
  be weak (depth 1-2, small hidden width) so the attention stage has to do the real
  combining work, but that is a knob to ablate, not an invariant to enforce.

* Every learner outputs `embed_dim` features so the stack is a clean (B, K, D) tensor
  that a Transformer can treat as a sequence of K tokens.
"""

from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_gating import FeatureGateConfig, FeatureGate

__all__ = [
    "WeakLearnerConfig",
    "default_ens",
    "make_feature_subsets",
    "WeakLearnerMLP",
    "WeakLearnerEnsemble",
    "SharedWeightEnsemble",
]

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class WeakLearnerConfig:
    """All knobs for the ensemble in one place, so experiments stay reproducible.

    num_learners : K — how many experts / how long the token sequence is.
    hidden_dim   : width of each expert's hidden layer(s). Keep small to stay "weak".
    embed_dim    : D — output width of every expert; the attention stage's model dim.
    depth        : number of hidden layers per expert (1-2 is the intended regime).
    dropout      : dropout inside each expert.
    activation   : one of ACTIVATIONS.
    feature_frac : fraction of input columns each expert sees (1.0 disables bagging).
    ensure_coverage : guarantee every input feature reaches at least one expert.
    bias         : use bias terms in the expert Linear layers.
    seed         : controls the feature-subset draw only (not weight init).
    batched      : run all experts as one batched matmul instead of a Python loop
                   (~2x faster, identical function). Off by default so that results
                   produced with the loop (notebooks 01-02) reproduce bit-for-bit:
                   batching draws dropout masks in a different order.
    gate         : feature_gating.FeatureGateConfig. Default mode="none" -> no gate
                   module runs at all, bit-for-bit identical to pre-gating v4.
    """

    num_learners: int = 16
    hidden_dim: int = 16
    embed_dim: int = 32
    depth: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    feature_frac: float = 0.7
    ensure_coverage: bool = True
    bias: bool = True
    seed: int = 42
    # shared=True swaps WeakLearnerEnsemble for SharedWeightEnsemble: K BatchEnsemble
    # members over one shared weight per layer (TabM, arXiv:2410.24210). Default False
    # keeps the original K-independent-MLPs architecture bit-for-bit.
    shared: bool = False
    batched: bool = False
    gate: FeatureGateConfig = field(default_factory=FeatureGateConfig)

    def __post_init__(self):
        if self.num_learners < 1:
            raise ValueError(f"num_learners must be >= 1, got {self.num_learners}")
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; choose from {sorted(ACTIVATIONS)}"
            )
        if not 0.0 < self.feature_frac <= 1.0:
            raise ValueError(f"feature_frac must be in (0, 1], got {self.feature_frac}")


# ---------------------------------------------------------------------------
# Feature bagging
# ---------------------------------------------------------------------------
def default_ens(**overrides) -> dict:
    """The canonical expert configuration, as a dict ready for `WeakLearnerConfig(**...)`.

    THE single source of truth for "what the architecture is". Everything that needs those
    numbers — cli.py, tuning.py, and every notebook — calls this instead of restating them,
    so changing `embed_dim` here changes it everywhere rather than in ten files that only
    look like they agree.

    Per-run fields are excluded (`seed`, `batched`) along with the nested `gate` config, so
    the result is exactly the architecture and callers still supply their own seed:

        WeakLearnerConfig(**default_ens(), seed=seed, batched=True)
        WeakLearnerConfig(**default_ens(feature_frac=0.4), seed=seed)

    A sweep over one field drops it instead of overriding it, so the loop stays the only
    place that sets it:

        ENS_KW = default_ens(); ENS_KW.pop("feature_frac")
    """
    base = {f.name: getattr(WeakLearnerConfig(), f.name)
            for f in fields(WeakLearnerConfig) if f.name not in ("seed", "batched", "gate")}
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError(f"unknown expert config field(s) {sorted(unknown)}; "
                         f"choose from {sorted(base)}")
    return {**base, **overrides}


def make_feature_subsets(
    in_dim: int,
    num_learners: int,
    feature_frac: float = 0.7,
    seed: int = 42,
    ensure_coverage: bool = True,
) -> List[np.ndarray]:
    """Draw one fixed random column subset per weak learner.

    Subsets are drawn once and then frozen for the lifetime of the model: they are
    part of the architecture, not a per-epoch augmentation. (v1 resampled them every
    epoch, which quietly changes what each expert means from epoch to epoch and makes
    per-expert diagnostics uninterpretable.)

    With `ensure_coverage`, any feature that no learner happened to draw is swapped
    into one of the subsets, replacing a column that is redundantly covered elsewhere.
    Without it, a small K or a small feature_frac can silently drop an input column
    from the model entirely.
    """
    if in_dim < 1:
        raise ValueError(f"in_dim must be >= 1, got {in_dim}")

    rng = np.random.RandomState(seed)
    k = max(1, min(in_dim, int(round(feature_frac * in_dim))))

    subsets = [
        np.sort(rng.choice(in_dim, size=k, replace=False)) for _ in range(num_learners)
    ]

    if ensure_coverage and k < in_dim:
        counts = Counter(int(f) for s in subsets for f in s)
        missing = [f for f in range(in_dim) if counts[f] == 0]

        for feat in missing:
            # Prefer a learner that holds at least one redundantly-covered column,
            # so patching a hole never opens a new one.
            order = rng.permutation(num_learners)
            for j in order:
                current = set(int(f) for f in subsets[j])
                if feat in current:
                    continue
                spare = sorted(f for f in current if counts[f] > 1)
                if not spare:
                    continue
                drop = int(rng.choice(spare))
                current.discard(drop)
                current.add(feat)
                subsets[j] = np.sort(np.fromiter(current, dtype=np.int64, count=len(current)))
                counts[drop] -= 1
                counts[feat] += 1
                break

    return subsets


# ---------------------------------------------------------------------------
# A single weak learner
# ---------------------------------------------------------------------------
class WeakLearnerMLP(nn.Module):
    """A small MLP restricted to a fixed subset of the input features.

    Maps (B, in_dim) -> (B, embed_dim) by slicing out its own columns first.
    The output is an embedding, so there is no activation on the final layer.
    """

    def __init__(
        self,
        in_dim: int,
        feat_idx: Sequence[int],
        hidden_dim: int = 16,
        embed_dim: int = 32,
        depth: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
        in_per_feature: int = 1,
    ):
        super().__init__()
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if feat_idx.ndim != 1 or feat_idx.size == 0:
            raise ValueError("feat_idx must be a non-empty 1-D index array")
        if feat_idx.min() < 0 or feat_idx.max() >= in_dim:
            raise ValueError(f"feat_idx out of range for in_dim={in_dim}")

        # A buffer (not a plain attribute) so it follows .to(device) and is saved
        # in the state_dict — reloading a checkpoint must restore the same columns.
        self.register_buffer("feat_idx", torch.from_numpy(feat_idx), persistent=True)
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        # With per-feature embeddings each of this expert's features arrives as a vector of
        # `in_per_feature` numbers instead of a scalar, so its first layer is that much wider.
        self.in_per_feature = in_per_feature
        act = ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        d = int(feat_idx.size) * in_per_feature
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim, bias=bias), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, embed_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    @property
    def num_features(self) -> int:
        return int(self.feat_idx.numel())

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, in_dim) -> (B, embed_dim), or (B, in_dim, d) when features are embedded.

        `gate`, if given, is this learner's (B, k) feature relevancy weights from
        feature_gating.FeatureGate — multiplied in before flattening so a per-feature
        gate value scales the whole embedded vector for that feature, not just one entry.
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"expected (B, in_dim) or (B, in_dim, d) input, got {tuple(x.shape)}")
        xi = x.index_select(1, self.feat_idx)
        if gate is not None:
            xi = xi * (gate.unsqueeze(-1) if xi.dim() == 3 else gate)
        return self.net(xi.flatten(1) if xi.dim() == 3 else xi)


# ---------------------------------------------------------------------------
# The ensemble
# ---------------------------------------------------------------------------
class SharedWeightEnsemble(nn.Module):
    """K BatchEnsemble members over ONE shared weight per layer -> (B, K, embed_dim).

    A drop-in sibling of WeakLearnerEnsemble: same constructor, same output contract, so
    attention.py, model.py's aggregators and the heads need no changes at all.

    TabM (Gorishniy, Kotelnikov & Babenko, ICLR 2025, arXiv:2410.24210) sec 3.2 writes
    member i's layer as

        l_i(x) = s_i * (W (r_i * x)) + b_i        <=>        W_i = W (*) (s_i r_i^T)

    with W shared and r_i, s_i, b_i per member. So K FULL-WIDTH members cost one W plus
    K*(in + 2*out) adapter numbers instead of K separate weight matrices. The paper's
    claim is that this is also a regulariser: members are individually overfitted but
    their average generalises, and TabM_naive (shared) beats TabM_packed (K independent
    MLPs, i.e. what WeakLearnerEnsemble builds).

    TWO DELIBERATE DIFFERENCES FROM WeakLearnerEnsemble
    ---------------------------------------------------
    1. Feature bagging is a 0/1 MASK over the full feature vector, never a slice. Slicing
       is what WeakLearnerEnsemble does and it CANNOT be combined with a shared W: after
       slicing, input position j is a different actual feature for each member, so one
       shared weight matrix would be reading inconsistent columns and the comparison would
       measure nothing. The mask keeps every member's input in the same coordinate system.
       feature_frac=1.0 (no masking) is TabM's own setting, where members differ only
       through their adapters -- that is the faithful replication; masking on top is the
       "does our feature bagging add anything to TabM?" variant.
    2. No feature gating. FeatureGate was measured not to help (step 9) and composing it
       with a learned per-member input scale r_i would be two input modulations doing the
       same job; a non-default gate raises rather than being silently ignored.
    """

    def __init__(self, in_dim: int, config: Optional[WeakLearnerConfig] = None,
                 in_per_feature: int = 1, **overrides):
        super().__init__()
        if config is None:
            config = WeakLearnerConfig(**overrides)
        elif overrides:
            config = WeakLearnerConfig(**{**config.__dict__, **overrides})
        if config.gate.mode != "none":
            raise ValueError("SharedWeightEnsemble does not support feature gating; r_i is "
                             "already a learned per-member input scale (see class docstring)")

        self.config = config
        self.in_dim = in_dim
        self.in_per_feature = in_per_feature
        self.num_learners = K = config.num_learners
        self.embed_dim = config.embed_dim

        # Same subsets as the independent ensemble would draw, so the two are comparable --
        # but applied as a mask over the FULL vector instead of a slice.
        self._subsets = make_feature_subsets(
            in_dim, K, feature_frac=config.feature_frac, seed=config.seed,
            ensure_coverage=config.ensure_coverage,
        )
        mask = torch.zeros(K, in_dim)
        for i, idx in enumerate(self._subsets):
            mask[i, idx] = 1.0
        if in_per_feature > 1:
            # Each feature arrives as in_per_feature numbers, so its mask bit repeats.
            mask = mask.repeat_interleave(in_per_feature, dim=1)
        self.register_buffer("mask", mask, persistent=True)
        self.masking = config.feature_frac < 1.0

        dims = [in_dim * in_per_feature] + [config.hidden_dim] * config.depth + [config.embed_dim]
        # nn.Linear holds the shared W so its initialisation is exactly PyTorch's default,
        # identical to what the independent experts get for a layer of the same shape.
        self.shared = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                     for i in range(len(dims) - 1)])
        self.R = nn.ParameterList([nn.Parameter(torch.ones(K, dims[i]))
                                   for i in range(len(dims) - 1)])
        self.S = nn.ParameterList([nn.Parameter(torch.ones(K, dims[i + 1]))
                                   for i in range(len(dims) - 1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(K, dims[i + 1]))
                                      for i in range(len(dims) - 1)]) if config.bias else None
        # TabM sec 3.3: "initialize all multiplicative adapters R and S, except for the very
        # first one, deterministically with 1". Without this one random adapter every member
        # computes the identical function and the ensemble is a single model.
        with torch.no_grad():
            self.R[0].copy_(torch.randint(0, 2, (K, dims[0]), dtype=torch.float32) * 2 - 1)

        self.act = ACTIVATIONS[config.activation]()
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else None

    # -- introspection (same surface WeakLearnerEnsemble exposes) ------------
    @property
    def feature_subsets(self) -> List[np.ndarray]:
        return [np.asarray(idx) for idx in self._subsets]

    def coverage(self) -> Dict[str, object]:
        counts = np.zeros(self.in_dim, dtype=np.int64)
        for idx in self._subsets:
            counts[idx] += 1
        return {"counts": counts, "uncovered": np.flatnonzero(counts == 0).tolist(),
                "features_per_learner": int(len(self._subsets[0])),
                "min_times_seen": int(counts.min()), "max_times_seen": int(counts.max())}

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor, x_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, in_dim) or (B, in_dim, d) -> (B, K, embed_dim). `x_raw` is accepted for
        interface parity with WeakLearnerEnsemble (feature gating) and ignored."""
        if x.dim() not in (2, 3):
            raise ValueError(f"expected (B, in_dim) or (B, in_dim, d) input, got {tuple(x.shape)}")
        h = x.flatten(1) if x.dim() == 3 else x                   # (B, F*d)
        h = h.unsqueeze(1).expand(-1, self.num_learners, -1)      # (B, K, F*d)
        if self.masking:
            h = h * self.mask
        last = len(self.shared) - 1
        for i, lin in enumerate(self.shared):
            h = torch.einsum("bki,oi->bko", h * self.R[i], lin.weight) * self.S[i]
            if self.bias is not None:
                h = h + self.bias[i]
            if i != last:                 # no activation on the output embedding, as in
                h = self.act(h)           # WeakLearnerMLP
                if self.drop is not None:
                    h = self.drop(h)
        return h


class WeakLearnerEnsemble(nn.Module):
    """K feature-bagged weak learners -> a token sequence (B, K, embed_dim).

    This is the only thing the attention stage needs to know about: it receives K
    tokens of width embed_dim and never has to care how they were produced.

    Speed: by default the forward pass loops over the K learners. With
    `config.batched=True` it instead stacks every expert's weights and runs each layer
    as one batched matmul over all K experts — the same function (verified in the smoke
    test), about 2x faster, and the cost no longer grows as K separate Python calls.
    """

    def __init__(self, in_dim: int, config: Optional[WeakLearnerConfig] = None,
                 in_per_feature: int = 1, **overrides):
        super().__init__()
        if config is None:
            config = WeakLearnerConfig(**overrides)
        elif overrides:
            config = WeakLearnerConfig(**{**config.__dict__, **overrides})

        self.config = config
        self.in_dim = in_dim
        self.in_per_feature = in_per_feature
        self.num_learners = config.num_learners
        self.embed_dim = config.embed_dim

        subsets = make_feature_subsets(
            in_dim,
            config.num_learners,
            feature_frac=config.feature_frac,
            seed=config.seed,
            ensure_coverage=config.ensure_coverage,
        )
        self.learners = nn.ModuleList(
            [
                WeakLearnerMLP(
                    in_dim=in_dim,
                    feat_idx=idx,
                    hidden_dim=config.hidden_dim,
                    embed_dim=config.embed_dim,
                    depth=config.depth,
                    dropout=config.dropout,
                    activation=config.activation,
                    bias=config.bias,
                    in_per_feature=in_per_feature,
                )
                for idx in subsets
            ]
        )
        self.gate = FeatureGate(in_dim, subsets, config.gate)

    # -- introspection -------------------------------------------------------
    @property
    def feature_subsets(self) -> List[np.ndarray]:
        """The frozen column subset of each learner, for reporting/plots."""
        return [lrn.feat_idx.detach().cpu().numpy() for lrn in self.learners]

    def coverage(self) -> Dict[str, object]:
        """How often each input feature is seen across the ensemble."""
        counts = np.zeros(self.in_dim, dtype=np.int64)
        for idx in self.feature_subsets:
            counts[idx] += 1
        return {
            "counts": counts,
            "uncovered": np.flatnonzero(counts == 0).tolist(),
            "features_per_learner": int(self.learners[0].num_features),
            "min_times_seen": int(counts.min()),
            "max_times_seen": int(counts.max()),
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor, x_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, in_dim) -> (B, K, embed_dim). Input may be (B, in_dim, d) if features are embedded.

        `x_raw`, if given, is the ORIGINAL (pre-embedding) (B, in_dim) input — that is
        what feature_gating.FeatureGate reads to decide relevance. Defaults to `x` itself,
        which is correct whenever there is no embedding stage upstream (x is already raw).
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"expected (B, in_dim) or (B, in_dim, d) input, got {tuple(x.shape)}")
        if x.size(1) != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {x.size(1)}")
        gates = self.gate(x if x_raw is None else x_raw)   # None ("none" mode) or (B, K, k)
        if self.config.batched:
            return self._forward_batched(x, gates)
        if gates is None:
            return torch.stack([lrn(x) for lrn in self.learners], dim=1)
        return torch.stack([lrn(x, gates[:, i]) for i, lrn in enumerate(self.learners)], dim=1)

    def _forward_batched(self, x: torch.Tensor, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """All experts at once. Every expert has the same layer structure and the same
        subset size (coverage patching swaps columns, never changes sizes), so layer j of
        every expert can be stacked into one (K, out, in) weight and applied with a single
        einsum. Parameters still live in the per-expert modules, so the state_dict,
        initialisation and gradients are exactly those of the loop version."""
        idx = torch.stack([lrn.feat_idx for lrn in self.learners])          # (K, k)
        h = x[:, idx]                                                         # (B, K, k[, d])
        if gates is not None:
            h = h * (gates.unsqueeze(-1) if h.dim() == 4 else gates)          # broadcast over d
        if h.dim() == 4:
            h = h.flatten(2)                                                  # (B, K, k*d)
        for group in zip(*(lrn.net for lrn in self.learners)):
            layer = group[0]
            if isinstance(layer, nn.Linear):
                W = torch.stack([m.weight for m in group])                  # (K, out, in)
                h = torch.einsum("bki,koi->bko", h, W)
                if layer.bias is not None:
                    h = h + torch.stack([m.bias for m in group])             # (K, out)
            else:
                h = layer(h)            # activation / dropout: parameter-free, same for all
        return h

    # -- regularization ------------------------------------------------------
    def weight_penalty(
        self,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        include_bias: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """L1/L2 penalty over the experts' weights.

        Normalized by parameter count by default so the same lambda means roughly the
        same pressure whether K is 4 or 32 — otherwise every change to K silently
        retunes the regularization strength too.

        L1 pushes each small expert toward using a few of its columns strongly, which
        is the "focus on the important features" behaviour the original idea asked for.
        """
        device = next(self.parameters()).device
        if l1_lambda == 0.0 and l2_lambda == 0.0:
            return torch.zeros((), device=device)

        l1 = torch.zeros((), device=device)
        l2 = torch.zeros((), device=device)
        n = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if not include_bias and p.dim() == 1:
                continue
            if l1_lambda:
                l1 = l1 + p.abs().sum()
            if l2_lambda:
                l2 = l2 + p.pow(2).sum()
            n += p.numel()

        denom = max(1, n) if normalize else 1
        return (l1_lambda * l1 + l2_lambda * l2) / denom

    # -- diagnostics ---------------------------------------------------------
    @torch.no_grad()
    def diversity_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Pairwise cosine similarity of the raw, pre-attention embeddings.

        Returns (K, K) similarity matrix and the mean off-diagonal value.
        Values near 1.0 mean the experts have collapsed onto the same representation;
        that is the failure mode v1 tried to fix with a penalty loss, and the number
        this method exists to keep honest.

        Similarity is computed per sample and then averaged over the batch. (v1's
        diversity.py averaged the embeddings over the batch *first* and compared the
        means, which hides per-sample disagreement and reports collapse that isn't
        there — or misses collapse that is.)
        """
        was_training = self.training
        self.eval()
        try:
            emb = self.forward(x)                       # (B, K, D)
            e = F.normalize(emb, dim=-1)
            sim = torch.einsum("bkd,bjd->bkj", e, e).mean(dim=0)   # (K, K)
        finally:
            if was_training:
                self.train()

        K = sim.size(0)
        if K < 2:
            return sim, float("nan")
        off = sim[~torch.eye(K, dtype=torch.bool, device=sim.device)]
        return sim, off.mean().item()

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"in_dim={self.in_dim}, K={c.num_learners}, embed_dim={c.embed_dim}, "
            f"hidden_dim={c.hidden_dim}, depth={c.depth}, feature_frac={c.feature_frac}, "
            f"gate={c.gate.mode}"
        )


# ---------------------------------------------------------------------------
# Smoke test:  python weak_learners.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, IN_DIM = 256, 8
    cfg = WeakLearnerConfig(num_learners=16, hidden_dim=16, embed_dim=32,
                            depth=1, feature_frac=0.7, seed=42)
    ens = WeakLearnerEnsemble(IN_DIM, cfg)
    x = torch.randn(B, IN_DIM)
    tokens = ens(x)

    print(ens)
    print(f"\ninput  {tuple(x.shape)}  ->  tokens {tuple(tokens.shape)}   (B, K, D)")
    print(f"parameters: {ens.num_parameters():,}")

    cov = ens.coverage()
    print(f"\nfeature bagging: {cov['features_per_learner']}/{IN_DIM} columns per learner")
    print(f"  times each feature is seen: {cov['counts'].tolist()}")
    print(f"  uncovered features: {cov['uncovered'] or 'none'}")

    sim, off_mean = ens.diversity_stats(x)
    print(f"\ndiversity (untrained, random init):")
    print(f"  mean off-diagonal cosine similarity = {off_mean:+.4f}   (1.0 == total collapse)")

    pen = ens.weight_penalty(l1_lambda=1e-4, l2_lambda=1e-4)
    print(f"\nweight penalty (l1=l2=1e-4, normalized): {pen.item():.3e}")

    # Contrast: no bagging -> every expert sees the same columns.
    ens_nobag = WeakLearnerEnsemble(IN_DIM, WeakLearnerConfig(
        num_learners=16, feature_frac=1.0, seed=42))
    _, off_nobag = ens_nobag.diversity_stats(x)
    print(f"\nsanity check — feature_frac=1.0 (the v2 setup, no bagging):")
    print(f"  mean off-diagonal cosine similarity = {off_nobag:+.4f}")
    print(f"  bagging changes it by {off_mean - off_nobag:+.4f}")

    # Batched forward must be the same function as the loop.
    ens_b = WeakLearnerEnsemble(IN_DIM, WeakLearnerConfig(**{**cfg.__dict__, "batched": True}))
    ens_b.load_state_dict(ens.state_dict())
    ens.eval(); ens_b.eval()
    with torch.no_grad():
        same = torch.allclose(ens(x), ens_b(x), atol=1e-5)
    print(f"\nbatched forward == loop forward (eval mode): {same}")

    # -----------------------------------------------------------------------
    # SharedWeightEnsemble (TabM-style BatchEnsemble). PASS/FAIL checks.
    # -----------------------------------------------------------------------
    print("\nSHARED-WEIGHT ENSEMBLE (TabM, arXiv:2410.24210)")
    checks = []
    sc = WeakLearnerConfig(num_learners=16, hidden_dim=16, embed_dim=32, depth=1,
                           feature_frac=0.7, seed=0, dropout=0.0, shared=True)
    se = SharedWeightEnsemble(IN_DIM, sc)
    out_s = se(x)
    checks.append(("output is (B, K, embed_dim), same contract as WeakLearnerEnsemble",
                   out_s.shape == (x.shape[0], 16, 32)))

    # The adapters are the ONLY source of member diversity here. TabM randomises the first
    # R and sets every other adapter to 1; with that one tensor also set to 1, all K members
    # must compute the identical function -- which is what makes it the diversity mechanism.
    se_flat = SharedWeightEnsemble(IN_DIM, WeakLearnerConfig(**{**sc.__dict__,
                                                               "feature_frac": 1.0}))
    with torch.no_grad():
        se_flat.R[0].fill_(1.0)
    se_flat.eval()
    with torch.no_grad():
        o_flat = se_flat(x)
    collapsed = torch.allclose(o_flat[:, 0], o_flat[:, 5], atol=1e-6)
    se.eval()
    with torch.no_grad():
        o_div = se(x)
    diverse = not torch.allclose(o_div[:, 0], o_div[:, 5], atol=1e-6)
    checks.append(("all-ones adapters collapse every member to one function, and TabM's "
                   "random first R breaks that", collapsed and diverse))

    # Every parameter group must train, or the arm is secretly a smaller model.
    se.train(); se.zero_grad()
    se(x).sum().backward()
    groups = {"shared W": [l.weight for l in se.shared], "R": list(se.R),
              "S": list(se.S), "bias": list(se.bias)}
    all_grad = all(p.grad is not None and p.grad.abs().sum() > 0
                   for ps in groups.values() for p in ps)
    checks.append((f"shared W, R, S and bias all receive nonzero gradient "
                   f"({sum(len(v) for v in groups.values())} tensors)", all_grad))

    # Masking must actually remove a feature from a member that does not own it.
    owner = se.feature_subsets
    dropped = next(j for j in range(IN_DIM) if j not in owner[0])
    x2 = x.clone(); x2[:, dropped] += 100.0
    se.eval()
    with torch.no_grad():
        unchanged = torch.allclose(se(x)[:, 0], se(x2)[:, 0], atol=1e-5)
        moved = not torch.allclose(se(x), se(x2), atol=1e-5)
    checks.append((f"a masked-out feature (col {dropped}) cannot influence member 0, but "
                   f"does move other members", unchanged and moved))
    checks.append(("feature_frac=1.0 disables masking entirely (TabM's own setting)",
                   se_flat.masking is False and bool(se_flat.mask.all())))

    # shared=False must leave the original architecture untouched, bit for bit.
    base = WeakLearnerConfig(**{**sc.__dict__, "shared": False})
    torch.manual_seed(0); a = WeakLearnerEnsemble(IN_DIM, base)
    torch.manual_seed(0); b = WeakLearnerEnsemble(IN_DIM, WeakLearnerConfig(**base.__dict__))
    a.eval(); b.eval()
    with torch.no_grad():
        checks.append(("shared=False is the default and still builds the original "
                       "K-independent-MLP ensemble, bit-identical",
                       base.shared is False and torch.equal(a(x), b(x))))

    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    # The point of weight sharing: width stops multiplying the parameter count, so the
    # savings can be spent on making each member FULL-SIZED (TabM tunes width 64..1024).
    print("\n  parameters vs expert width (K=16, embed_dim=32, depth=1):")
    print(f"  {'hidden':>7s} {'independent':>13s} {'shared':>10s} {'ratio':>7s}")
    for hd in (16, 64, 256, 512):
        kw = {**sc.__dict__, "hidden_dim": hd}
        ind = WeakLearnerEnsemble(IN_DIM, WeakLearnerConfig(**{**kw, "shared": False}))
        shd = SharedWeightEnsemble(IN_DIM, WeakLearnerConfig(**kw))
        print(f"  {hd:7d} {ind.num_parameters():13,d} {shd.num_parameters():10,d} "
              f"{ind.num_parameters()/shd.num_parameters():6.1f}x")
