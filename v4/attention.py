"""
attention.py  —  SRP v4, step 2

The sequencing stage. Takes the (B, K, D) token block produced by
`weak_learners.WeakLearnerEnsemble` and lets each expert read the experts before it.

    (B, K, D) independent opinions  ->  (B, K, D) context-aware opinions

This module makes NO predictions. It is a pure encoder: tokens in, tokens out.
The readout (mean-pool, last-token, boosted accumulation, ...) lives in the model
file so that the aggregation strategy can be swapped without touching the attention
mechanism, and so the causal property can be tested completely on its own.

Design notes
------------
* CAUSALITY IS THE WHOLE CLAIM, so it is tested, not asserted. The block at the
  bottom of this file proves it three ways: by perturbation, by gradient, and by
  inspecting the attention matrix. If token i can see token j>i, those tests fail.

* `causal` IS A FLAG. The v3 post-mortem's open question — "does the ordering
  actually matter, or would cheap bidirectional pooling do as well?" — is a
  one-line ablation only if bidirectional is a supported mode from the start.

* PRE-NORM + RESIDUAL. v2 found (correctly) that these are prerequisites for
  training more than one block without vanishing gradients. Kept verbatim.

* THE MASK IS BUILT ONCE AND CACHED. v2 rebuilt a (K, K) -inf matrix inside every
  block on every forward pass — allocation churn for a constant.

* STEP EMBEDDINGS LIVE HERE. Without them the causal mask is the only thing telling
  the stage that experts are ordered, and position 3 is otherwise interchangeable
  with position 7 apart from how many neighbours it can see.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AttentionConfig",
    "build_causal_mask",
    "CausalSelfAttentionBlock",
    "CausalAttentionStage",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AttentionConfig:
    """Knobs for the sequencing stage.

    embed_dim          : D — must match the ensemble's embed_dim and divide by num_heads.
    num_heads          : attention heads per block.
    attn_depth         : how many stacked blocks.
    ff_mult            : feed-forward width = embed_dim * ff_mult.
    dropout            : applied to attention output, FF, and residual adds.
    causal             : True  -> expert i sees experts 1..i  (the architecture's claim)
                         False -> every expert sees every other (ablation control)
    use_step_embedding : add a learnable per-position vector before the blocks.
    need_weights       : return attention matrices for diagnostics. Costs a little
                         speed; turn off for production training runs.
    """

    embed_dim: int = 32
    num_heads: int = 4
    attn_depth: int = 2
    ff_mult: int = 4
    dropout: float = 0.1
    causal: bool = True
    use_step_embedding: bool = True
    need_weights: bool = True

    def __post_init__(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.attn_depth < 1:
            raise ValueError(f"attn_depth must be >= 1, got {self.attn_depth}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.ff_mult < 1:
            raise ValueError(f"ff_mult must be >= 1, got {self.ff_mult}")


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------
def build_causal_mask(seq_len: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Additive attention mask: 0 where attending is allowed, -inf where forbidden.

    `triu(diagonal=1)` keeps the diagonal and everything below it at 0, so row i
    (the query at position i) can attend to columns 0..i inclusive — an expert sees
    itself and everyone before it. Row 0 keeps exactly one open column, so no row is
    ever fully masked (a fully masked row would softmax to NaN).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


# ---------------------------------------------------------------------------
# One block
# ---------------------------------------------------------------------------
class CausalSelfAttentionBlock(nn.Module):
    """Pre-norm residual self-attention + feed-forward.

        x = x + Dropout(Attn(LN(x)))
        x = x + Dropout(FF(LN(x)))

    Pre-norm (normalising the *input* of each sublayer rather than the output of the
    residual add) is what keeps a clean gradient path from the last block back to the
    weak learners.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = embed_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.norm1(x)
        attn_out, attn_w = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=True,      # (B, K, K), averaged over heads
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_w


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
class CausalAttentionStage(nn.Module):
    """Stack of causal blocks over weak-learner tokens.

    forward:  (B, K, D)  ->  (B, K, D),  attention weights per block

    The output at position i is a context-aware version of expert i's opinion: it has
    absorbed experts 0..i and nothing after. Position K-1 is the only one that has
    seen the whole ensemble.
    """

    def __init__(self, config: Optional[AttentionConfig] = None,
                 max_len: int = 64, **overrides):
        super().__init__()
        if config is None:
            config = AttentionConfig(**overrides)
        elif overrides:
            config = AttentionConfig(**{**config.__dict__, **overrides})

        self.config = config
        self.embed_dim = config.embed_dim
        self.max_len = max_len

        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_step_embedding:
            self.step_embedding = nn.Parameter(torch.zeros(1, max_len, config.embed_dim))
            nn.init.trunc_normal_(self.step_embedding, std=0.02)
        else:
            self.register_parameter("step_embedding", None)

        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(
                config.embed_dim, config.num_heads, config.dropout, config.ff_mult
            )
            for _ in range(config.attn_depth)
        ])
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Cache masks by sequence length; they are constants, not parameters.
        self._mask_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

    # -- mask ---------------------------------------------------------------
    def causal_mask(self, seq_len: int, device, dtype) -> Optional[torch.Tensor]:
        """Cached additive mask, or None when running in bidirectional (ablation) mode."""
        if not self.config.causal:
            return None
        key = (seq_len, str(device), str(dtype))
        if key not in self._mask_cache:
            self._mask_cache[key] = build_causal_mask(seq_len, device=device, dtype=dtype)
        return self._mask_cache[key]

    # -- forward ------------------------------------------------------------
    def forward(self, tokens: torch.Tensor,
                need_weights: Optional[bool] = None
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if tokens.dim() != 3:
            raise ValueError(f"expected (B, K, D) tokens, got {tuple(tokens.shape)}")
        B, K, D = tokens.shape
        if D != self.embed_dim:
            raise ValueError(f"expected embed_dim={self.embed_dim}, got {D}")
        if self.step_embedding is not None and K > self.max_len:
            raise ValueError(
                f"sequence length {K} exceeds max_len={self.max_len}; "
                f"construct the stage with max_len >= {K}"
            )

        if need_weights is None:
            need_weights = self.config.need_weights

        h = self.input_norm(tokens)
        if self.step_embedding is not None:
            h = h + self.step_embedding[:, :K]
        h = self.dropout(h)

        mask = self.causal_mask(K, tokens.device, h.dtype)

        attn_weights: List[torch.Tensor] = []
        for block in self.blocks:
            h, w = block(h, attn_mask=mask, need_weights=need_weights)
            if w is not None:
                attn_weights.append(w)

        return self.final_norm(h), (attn_weights or None)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        c = self.config
        mode = "causal" if c.causal else "bidirectional (ablation)"
        return (f"embed_dim={c.embed_dim}, heads={c.num_heads}, depth={c.attn_depth}, "
                f"ff_mult={c.ff_mult}, mode={mode}")


# ---------------------------------------------------------------------------
# Verification:  python attention.py
# ---------------------------------------------------------------------------
def _check_causality_by_perturbation(stage, B=4, K=8, D=32, tol=0.0):
    """Change token j only. Outputs at positions i < j must be bit-identical."""
    stage.eval()                                   # dropout off -> deterministic
    torch.manual_seed(0)
    x = torch.randn(B, K, D)
    with torch.no_grad():
        base, _ = stage(x)

    violations, expected_changes = [], []
    for j in range(K):
        xp = x.clone()
        xp[:, j] += 10.0                           # a large, unmistakable perturbation
        with torch.no_grad():
            out, _ = stage(xp)
        delta = (out - base).abs().amax(dim=(0, 2))   # max change at each position
        for i in range(K):
            if i < j and delta[i].item() > tol:
                violations.append((i, j, delta[i].item()))
            if i >= j and delta[i].item() <= 1e-9:
                expected_changes.append((i, j))
    return violations, expected_changes


def _check_causality_by_gradient(stage, K=8, D=32):
    """d(output_i)/d(input_j) must be exactly zero for j > i."""
    stage.eval()
    torch.manual_seed(0)
    x = torch.randn(1, K, D, requires_grad=True)
    out, _ = stage(x)

    leaks = []
    for i in range(K):
        if x.grad is not None:
            x.grad = None
        out[0, i].sum().backward(retain_graph=True)
        g = x.grad[0].abs().sum(dim=1)             # (K,) influence of each input token
        for j in range(i + 1, K):
            if g[j].item() > 0.0:
                leaks.append((i, j, g[j].item()))
    return leaks


if __name__ == "__main__":
    from weak_learners import WeakLearnerEnsemble, WeakLearnerConfig

    torch.manual_seed(0)
    B, IN_DIM, K, D = 64, 8, 8, 32

    cfg = AttentionConfig(embed_dim=D, num_heads=4, attn_depth=2, causal=True)
    stage = CausalAttentionStage(cfg)
    print(stage)
    print(f"\nattention stage parameters: {stage.num_parameters():,}")

    # ---- end-to-end wiring with step 1 ----------------------------------
    ens = WeakLearnerEnsemble(IN_DIM, WeakLearnerConfig(
        num_learners=K, hidden_dim=16, embed_dim=D, depth=1, feature_frac=0.7))
    x = torch.randn(B, IN_DIM)
    tokens = ens(x)
    ctx, attn = stage(tokens)
    print(f"\nwiring:  x {tuple(x.shape)} -> tokens {tuple(tokens.shape)} -> context {tuple(ctx.shape)}")
    print(f"attention maps returned: {len(attn)} (one per block), each {tuple(attn[0].shape)}")

    # ---- 1. attention matrix must be lower-triangular --------------------
    print("\n" + "=" * 68)
    print("TEST 1 — attention matrix is lower-triangular")
    stage.eval()
    with torch.no_grad():
        _, attn = stage(tokens)
    A = attn[0].mean(dim=0)                        # (K, K) averaged over batch
    upper = A[torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)]
    print(f"  max weight in the forbidden upper triangle : {upper.abs().max().item():.3e}")
    print(f"  each row sums to 1                         : "
          f"{torch.allclose(A.sum(dim=1), torch.ones(K), atol=1e-5)}")
    print(f"  -> {'PASS' if upper.abs().max().item() == 0.0 else 'FAIL'}")
    print("\n  row-by-row (how many experts each one may read):")
    for i in range(K):
        bar = "".join("#" if A[i, j] > 1e-6 else "." for j in range(K))
        print(f"    expert {i}: [{bar}]  sees {int((A[i] > 1e-6).sum())} of {K}")

    # ---- 2. perturbation ------------------------------------------------
    print("\n" + "=" * 68)
    print("TEST 2 — perturbing expert j leaves every earlier expert untouched")
    viol, missing = _check_causality_by_perturbation(stage, B=4, K=K, D=D)
    print(f"  causality violations (future leaked into past) : {len(viol)}")
    print(f"  positions that should have changed but didn't  : {len(missing)}")
    print(f"  -> {'PASS' if not viol else 'FAIL: ' + str(viol[:3])}")

    # ---- 3. gradient ----------------------------------------------------
    print("\n" + "=" * 68)
    print("TEST 3 — gradient of output_i w.r.t. input_j is exactly 0 for j > i")
    leaks = _check_causality_by_gradient(stage, K=K, D=D)
    print(f"  non-zero future gradients: {len(leaks)}")
    print(f"  -> {'PASS' if not leaks else 'FAIL: ' + str(leaks[:3])}")

    # ---- 4. the ablation flag must actually change behaviour ------------
    print("\n" + "=" * 68)
    print("TEST 4 — causal=False really does remove the constraint")
    bi = CausalAttentionStage(AttentionConfig(embed_dim=D, num_heads=4,
                                              attn_depth=2, causal=False))
    bi_viol, _ = _check_causality_by_perturbation(bi, B=4, K=K, D=D)
    bi.eval()
    with torch.no_grad():
        _, bi_attn = bi(tokens)
    bi_upper = bi_attn[0].mean(dim=0)[
        torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)]
    print(f"  bidirectional: upper-triangle mass = {bi_upper.abs().max().item():.4f} (should be > 0)")
    print(f"  bidirectional: 'violations' found  = {len(bi_viol)} (should be > 0 — no constraint)")
    print(f"  -> {'PASS' if len(bi_viol) > 0 and bi_upper.abs().max().item() > 0 else 'FAIL'}")

    # ---- 5. gradients reach the weak learners ---------------------------
    print("\n" + "=" * 68)
    print("TEST 5 — gradients flow back through the stage into the weak learners")
    ens.zero_grad(); stage.zero_grad(); stage.train()
    ctx, _ = stage(ens(x))
    ctx.mean().backward()
    gn = [(n, p.grad.abs().sum().item()) for n, p in ens.named_parameters() if p.grad is not None]
    dead = [n for n, g in gn if g == 0.0]
    print(f"  weak-learner tensors receiving gradient : {len(gn) - len(dead)}/{len(gn)}")
    print(f"  -> {'PASS' if not dead else 'FAIL: dead ' + str(dead[:3])}")
    print("=" * 68)
