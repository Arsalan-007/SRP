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

In that paper, adding these embeddings to a plain MLP improved it more than any
architecture change: Higgs 0.720 -> 0.728, Santander 0.912 -> 0.924, Facebook 5.686 -> 5.525.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["EmbeddingConfig", "FeatureEmbedding", "EMBEDDING_MODES"]

EMBEDDING_MODES = ("none", "linear", "periodic")


@dataclass
class EmbeddingConfig:
    """mode        : one of EMBEDDING_MODES.
    d_embedding : width of each feature's own vector (ignored for "none").
    n_frequencies: number of periodic frequencies per feature ("periodic" only).
    sigma       : std of the frequency initialisation. Larger = higher frequencies =
                  sharper functions of a single feature, but harder to optimise.
    """

    mode: str = "none"
    d_embedding: int = 8
    n_frequencies: int = 16
    sigma: float = 0.1

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

    @property
    def out_per_feature(self) -> int:
        """Numbers produced per input feature: 1 for "none", else d_embedding."""
        return 1 if self.mode == "none" else self.d_embedding


class FeatureEmbedding(nn.Module):
    """(B, F) -> (B, F, d), every feature encoded by its own parameters.

    Per-feature weights are held as (F, ...) tensors and applied with einsum, so feature j
    can never influence feature i's embedding. That is what breaks rotation invariance:
    after a rotation, "feature j" is a different mixture and these weights no longer match it.
    """

    def __init__(self, n_features: int, config: Optional[EmbeddingConfig] = None, **overrides):
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
        extra = f", n_frequencies={c.n_frequencies}, sigma={c.sigma}" if c.mode == "periodic" else ""
        return f"mode={c.mode}, d_embedding={c.d_embedding}{extra}, n_features={self.n_features}"


if __name__ == "__main__":
    torch.manual_seed(0)
    B, F = 64, 8
    x = torch.randn(B, F)

    print("SHAPES AND SIZES")
    for mode in EMBEDDING_MODES:
        emb = FeatureEmbedding(F, EmbeddingConfig(mode=mode, d_embedding=8, n_frequencies=16))
        out = emb(x)
        print(f"  {mode:9s} {tuple(x.shape)} -> {tuple(out.shape)}   params {emb.num_parameters():,}")

    print("\nCHECKS")
    emb = FeatureEmbedding(F, EmbeddingConfig(mode="none"))
    print(f"  [{'PASS' if torch.equal(emb(x), x) else 'FAIL'}] mode='none' is the identity (old path unchanged)")

    # Feature independence: change feature 3 only; no other feature's embedding may move.
    emb = FeatureEmbedding(F, EmbeddingConfig(mode="periodic")).eval()
    with torch.no_grad():
        a = emb(x)
        xp = x.clone(); xp[:, 3] += 1.0
        b = emb(xp)
    moved = [j for j in range(F) if not torch.allclose(a[:, j], b[:, j], atol=1e-6)]
    print(f"  [{'PASS' if moved == [3] else 'FAIL'}] only the changed feature's embedding moves (moved: {moved})")

    # Rotation sensitivity: the property the diagnostics said we lack.
    Q, _ = torch.linalg.qr(torch.randn(F, F))
    with torch.no_grad():
        plain_same = torch.allclose((x @ Q) @ Q.T, x, atol=1e-5)
        emb_rot = emb(x @ Q)
    print(f"  [{'PASS' if plain_same else 'FAIL'}] rotation is information-preserving for a Linear layer")
    print(f"  [{'PASS' if not torch.allclose(emb_rot, a, atol=1e-4) else 'FAIL'}] but the embedding of rotated data "
          f"differs -> the model can now tell the axes apart")

    # Sharpness: a periodic embedding can change fast in one feature; a linear one cannot.
    grid = torch.zeros(400, F); grid[:, 0] = torch.linspace(-2, 2, 400)
    with torch.no_grad():
        for mode in ("linear", "periodic"):
            e = FeatureEmbedding(F, EmbeddingConfig(mode=mode, d_embedding=8, n_frequencies=16, sigma=1.0)).eval()
            out = e(grid)[:, 0, :]
            print(f"  {mode:9s} max |change| between neighbouring x values: {out.diff(dim=0).abs().max():.4f}")
