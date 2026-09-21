"""
embeddings.py  —  SRP v4, step 5: per-feature numerical embeddings

WHY THIS EXISTS
---------------
The Grinsztajn diagnostics (notebook 04) showed our models are *rotation invariant*:
randomly rotating the features costs XGBoost up to 17 accuracy points and costs us
essentially nothing. A model that does not notice a rotation is a model that is not using
the individual feature axes — and on Poker and HIGGS, once the axes were scrambled, our
models actually BEAT XGBoost. So the axes are where the tree advantage lives, and we
cannot see them.

The reason is structural: every expert's first layer is a Linear over its columns, which
mixes them immediately. This module inserts a stage *before* any mixing, where each
feature is encoded on its own:

    x (B, F)  ->  embedding  ->  (B, F, d)      each feature gets its own d-vector
                                                 using its own parameters

Modes
-----
"none"      identity; x stays (B, F). The original v4 behaviour, bit-for-bit.
"linear"    per-feature Linear(1 -> d) + ReLU. The control: per-feature processing, but
            still a smooth, linear-in-x function. Tells us whether any gain comes from
            *per-feature* treatment or from the periodic part below.
"periodic"  per-feature Periodic -> Linear -> ReLU ("PLR", Gorishniy et al. 2022,
            "On Embeddings for Numerical Features in Tabular Deep Learning"):
                v(x) = [sin(2*pi*c_1*x) ... sin(2*pi*c_k*x),
                        cos(2*pi*c_1*x) ... cos(2*pi*c_k*x)]
            with learnable per-feature frequencies c initialised from N(0, sigma^2), then
            a per-feature Linear(2k -> d) and ReLU. High frequencies let the network
            represent sharp, threshold-like changes in a single feature, which is the
            second thing trees do better (irregular functions).
"piecewise_linear"  per-feature quantile-BINNED encoding ("PLE" in the same paper), then the
            same Linear -> ReLU tail. Fixed, non-learned bin edges (T+1 of them, from the
            TRAINING quantiles of each feature) instead of periodic's learned frequencies:
                v_t(x) = clip((x - edge[t]) / (edge[t+1] - edge[t]), 0, 1),  t = 0..T-1
            1.0 for every bin fully below x, 0.0 for every bin above, and the fractional
            position within x's own bin — the network gets an explicit "which bucket, how
            far into it" signal instead of having to construct one from smooth basis
            functions. Targets the SAME "irregular functions" property as "periodic", by a
            different mechanism (discretization vs. periodicity) — built specifically to
            answer whether periodic's win is about periodicity or just about any per-feature
            nonlinear treatment, after PLR measurably REGRESSED on YearPredictionMSD
            mean-pool (step 7). Needs `bin_edges` computed from training data BEFORE
            construction — see `compute_quantile_bins()`.

In the periodic-embeddings paper, adding PLR to a plain MLP improved it more than any
architecture change: Higgs 0.720 -> 0.728, Santander 0.912 -> 0.924, Facebook 5.686 -> 5.525.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = ["EmbeddingConfig", "FeatureEmbedding", "EMBEDDING_MODES", "compute_quantile_bins"]

EMBEDDING_MODES = ("none", "linear", "periodic", "piecewise_linear")


@dataclass
class EmbeddingConfig:
    """mode        : one of EMBEDDING_MODES.
    d_embedding : width of each feature's own vector (ignored for "none").
    n_frequencies: number of periodic frequencies per feature ("periodic" only).
    sigma       : std of the frequency initialisation. Larger = higher frequencies =
                  sharper functions of a single feature, but harder to optimise.
    n_bins      : number of quantile bins per feature ("piecewise_linear" only).
    """

    mode: str = "none"
    d_embedding: int = 8
    n_frequencies: int = 16
    sigma: float = 0.1
    n_bins: int = 16

    def __post_init__(self):
        if self.mode not in EMBEDDING_MODES:
            raise ValueError(f"mode must be one of {EMBEDDING_MODES}, got {self.mode!r}")
        if self.mode != "none":
            if self.d_embedding < 1:
                raise ValueError("d_embedding must be >= 1")
            if self.mode == "periodic" and self.n_frequencies < 1:
                raise ValueError("n_frequencies must be >= 1")
            if self.mode == "periodic" and self.sigma <= 0:
                raise ValueError("sigma must be > 0")
            if self.mode == "piecewise_linear" and self.n_bins < 2:
                raise ValueError("n_bins must be >= 2")

    @property
    def out_per_feature(self) -> int:
        """Numbers produced per input feature: 1 for "none", else d_embedding."""
        return 1 if self.mode == "none" else self.d_embedding


def compute_quantile_bins(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-feature bin edges from TRAINING data quantiles: (F, n_bins+1), strictly increasing.

    Fit on train only, like every other preprocessing step in this project (QuantileTransformer
    in datasets.py/benchmarks*.py). A constant or near-constant column would otherwise produce
    repeated quantile values (zero-width bins, division by zero in the encoding) — a tiny
    strictly-increasing nudge is added after `np.maximum.accumulate` to guarantee every bin has
    positive width regardless of how degenerate the column's distribution is.
    """
    X = np.asarray(X, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(X, qs, axis=0).T                       # (F, n_bins+1)
    edges = np.maximum.accumulate(edges, axis=1)
    # Tie-breaking offset scaled to each feature's OWN range (an absolute 1e-6 would vanish
    # under float32 rounding for large-magnitude features, and be needlessly coarse for
    # already-normalized ones). A fully-constant column falls back to a fixed small ladder.
    spread = edges[:, -1:] - edges[:, :1]
    spread = np.where(spread > 0, spread, 1.0)
    edges = edges + np.arange(n_bins + 1) * (1e-6 * spread)
    return edges.astype(np.float32)


class FeatureEmbedding(nn.Module):
    """(B, F) -> (B, F, d), every feature encoded by its own parameters.

    Per-feature weights are held as (F, ...) tensors and applied with einsum, so feature j
    can never influence feature i's embedding. That is what breaks rotation invariance:
    after a rotation, "feature j" is a different mixture and these weights no longer match it.
    """

    def __init__(self, n_features: int, config: Optional[EmbeddingConfig] = None,
                 bin_edges: Optional[np.ndarray] = None, **overrides):
        super().__init__()
        if config is None:
            config = EmbeddingConfig(**overrides)
        elif overrides:
            config = EmbeddingConfig(**{**config.__dict__, **overrides})
        self.config = config
        self.n_features = n_features
        d = config.d_embedding

        if config.mode == "none":
            return

        if config.mode == "periodic":
            # one set of frequencies per feature; learnable
            self.frequencies = nn.Parameter(torch.randn(n_features, config.n_frequencies) * config.sigma)
            in_dim = 2 * config.n_frequencies
        elif config.mode == "piecewise_linear":
            if bin_edges is None:
                raise ValueError(
                    "mode='piecewise_linear' needs bin_edges from compute_quantile_bins(X_train, "
                    "n_bins) — fixed, fit-from-data bin boundaries, not a learned parameter."
                )
            bin_edges = np.asarray(bin_edges, dtype=np.float32)
            if bin_edges.shape != (n_features, config.n_bins + 1):
                raise ValueError(f"expected bin_edges of shape ({n_features}, {config.n_bins + 1}), "
                                 f"got {bin_edges.shape}")
            # Frozen, not learned — a buffer, like weak_learners.py's feature subsets.
            self.register_buffer("bin_edges", torch.from_numpy(bin_edges), persistent=True)
            in_dim = config.n_bins
        else:
            in_dim = 1

        # per-feature Linear(in_dim -> d): weights (F, in_dim, d), bias (F, d)
        self.weight = nn.Parameter(torch.empty(n_features, in_dim, d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        self.activation = nn.ReLU()

    @property
    def out_per_feature(self) -> int:
        return self.config.out_per_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"expected (B, F) input, got {tuple(x.shape)}")
        if x.size(1) != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.size(1)}")
        if self.config.mode == "none":
            return x                                     # (B, F) — unchanged path

        if self.config.mode == "periodic":
            v = 2 * math.pi * self.frequencies.unsqueeze(0) * x.unsqueeze(-1)   # (B, F, k)
            h = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)                 # (B, F, 2k)
        elif self.config.mode == "piecewise_linear":
            lo = self.bin_edges[:, :-1].unsqueeze(0)                            # (1, F, T)
            hi = self.bin_edges[:, 1:].unsqueeze(0)                             # (1, F, T)
            h = ((x.unsqueeze(-1) - lo) / (hi - lo)).clamp(0.0, 1.0)             # (B, F, T)
        else:
            h = x.unsqueeze(-1)                                                 # (B, F, 1)

        h = torch.einsum("bfi,fio->bfo", h, self.weight) + self.bias            # (B, F, d)
        return self.activation(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        c = self.config
        if c.mode == "none":
            return "mode=none (identity)"
        if c.mode == "periodic":
            extra = f", n_frequencies={c.n_frequencies}, sigma={c.sigma}"
        elif c.mode == "piecewise_linear":
            extra = f", n_bins={c.n_bins}"
        else:
            extra = ""
        return f"mode={c.mode}, d_embedding={c.d_embedding}{extra}, n_features={self.n_features}"


if __name__ == "__main__":
    torch.manual_seed(0)
    B, F, N_BINS = 64, 8, 16
    x = torch.randn(B, F)
    x_train_np = np.random.RandomState(0).randn(2000, F).astype(np.float32)
    bins = compute_quantile_bins(x_train_np, N_BINS)

    def make(mode, **kw):
        if mode == "piecewise_linear":
            return FeatureEmbedding(F, EmbeddingConfig(mode=mode, n_bins=N_BINS, **kw), bin_edges=bins)
        return FeatureEmbedding(F, EmbeddingConfig(mode=mode, d_embedding=8, n_frequencies=16, **kw))

    print("SHAPES AND SIZES")
    for mode in EMBEDDING_MODES:
        emb = make(mode)
        out = emb(x)
        print(f"  {mode:16s} {tuple(x.shape)} -> {tuple(out.shape)}   params {emb.num_parameters():,}")

    print("\nCHECKS")
    emb = FeatureEmbedding(F, EmbeddingConfig(mode="none"))
    print(f"  [{'PASS' if torch.equal(emb(x), x) else 'FAIL'}] mode='none' is the identity (old path unchanged)")

    print("\ncompute_quantile_bins()")
    ok = bins.shape == (F, N_BINS + 1) and (np.diff(bins, axis=1) > 0).all()
    print(f"  [{'PASS' if ok else 'FAIL'}] shape ({F}, {N_BINS+1}) and every bin strictly positive width")
    const_col = np.zeros((500, 1), dtype=np.float32)
    const_bins = compute_quantile_bins(const_col, 4)
    ok = (np.diff(const_bins, axis=1) > 0).all()
    print(f"  [{'PASS' if ok else 'FAIL'}] a fully-constant column still gets strictly-increasing "
          f"edges (no division by zero): {const_bins[0].tolist()}")

    # Feature independence + rotation sensitivity, checked for BOTH "periodic" and the new
    # "piecewise_linear" mode -- the property that fixed the rotation-invariance diagnosis
    # (step 4) has to hold for any per-feature encoding, not just the periodic one.
    Q, _ = torch.linalg.qr(torch.randn(F, F))
    for mode in ("periodic", "piecewise_linear"):
        emb = make(mode).eval()
        with torch.no_grad():
            a = emb(x)
            xp = x.clone(); xp[:, 3] += 1.0
            b = emb(xp)
            emb_rot = emb(x @ Q)
        moved = [j for j in range(F) if not torch.allclose(a[:, j], b[:, j], atol=1e-6)]
        print(f"  [{'PASS' if moved == [3] else 'FAIL'}] {mode}: only the changed feature's embedding "
              f"moves (moved: {moved})")
        print(f"  [{'PASS' if not torch.allclose(emb_rot, a, atol=1e-4) else 'FAIL'}] {mode}: embedding "
              f"of rotated data differs -> the model can tell the axes apart")

    # Closed-form correctness: at a value exactly ON a known bin edge, every bin fully below
    # must read 1.0, every bin at-or-above must read 0.0 -- the defining property of PLE.
    emb = make("piecewise_linear").eval()
    t = 5
    x_on_edge = torch.zeros(1, F); x_on_edge[0, 0] = float(bins[0, t])
    with torch.no_grad():
        lo = emb.bin_edges[:, :-1]; hi = emb.bin_edges[:, 1:]
        raw = ((x_on_edge.unsqueeze(-1) - lo.unsqueeze(0)) / (hi - lo).unsqueeze(0)).clamp(0, 1)[0, 0]
    below_ok = torch.allclose(raw[:t], torch.ones(t), atol=1e-4)
    above_ok = torch.allclose(raw[t:], torch.zeros(N_BINS - t), atol=1e-4)
    print(f"  [{'PASS' if below_ok and above_ok else 'FAIL'}] at a bin edge, bins fully below read 1.0 "
          f"and bins at-or-above read 0.0 (raw encoding: {[round(v, 2) for v in raw.tolist()]})")

    # Sharpness: does binning also produce a sharp local reaction, like periodic does, unlike linear?
    grid = torch.zeros(400, F); grid[:, 0] = torch.linspace(-2, 2, 400)
    with torch.no_grad():
        for mode in ("linear", "periodic", "piecewise_linear"):
            kw = {"sigma": 1.0} if mode == "periodic" else {}
            e = make(mode, **kw).eval()
            out = e(grid)[:, 0, :]
            print(f"  {mode:16s} max |change| between neighbouring x values: {out.diff(dim=0).abs().max():.4f}")
