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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WeakLearnerConfig",
    "make_feature_subsets",
    "WeakLearnerMLP",
    "WeakLearnerEnsemble",
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
    batched: bool = False

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, embed_dim), or (B, in_dim, d) when features are embedded."""
        if x.dim() not in (2, 3):
            raise ValueError(f"expected (B, in_dim) or (B, in_dim, d) input, got {tuple(x.shape)}")
        xi = x.index_select(1, self.feat_idx)
        return self.net(xi.flatten(1) if xi.dim() == 3 else xi)


# ---------------------------------------------------------------------------
# The ensemble
# ---------------------------------------------------------------------------
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, K, embed_dim). Input may be (B, in_dim, d) if features are embedded."""
        if x.dim() not in (2, 3):
            raise ValueError(f"expected (B, in_dim) or (B, in_dim, d) input, got {tuple(x.shape)}")
        if x.size(1) != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {x.size(1)}")
        if self.config.batched:
            return self._forward_batched(x)
        return torch.stack([lrn(x) for lrn in self.learners], dim=1)

    def _forward_batched(self, x: torch.Tensor) -> torch.Tensor:
        """All experts at once. Every expert has the same layer structure and the same
        subset size (coverage patching swaps columns, never changes sizes), so layer j of
        every expert can be stacked into one (K, out, in) weight and applied with a single
        einsum. Parameters still live in the per-expert modules, so the state_dict,
        initialisation and gradients are exactly those of the loop version."""
        idx = torch.stack([lrn.feat_idx for lrn in self.learners])          # (K, k)
        h = x[:, idx]                                                         # (B, K, k[, d])
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
            f"hidden_dim={c.hidden_dim}, depth={c.depth}, feature_frac={c.feature_frac}"
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
