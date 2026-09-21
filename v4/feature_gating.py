"""
feature_gating.py  —  SRP v4, step 6: adaptive per-learner feature relevancy

WHY THIS EXISTS
---------------
Notebook 09 tuned every hyperparameter of the mean-pool + PLR arm on Covertype and
still recovered less than half the gap to XGBoost (0.9486 -> 0.9569, gap to XGBoost
0.0203 -> 0.0120). The notebook's own verdict: "MOSTLY NOT hyperparameters ... real
evidence for trying an architectural fix." This module is that fix.

Feature bagging (weak_learners.py) already gives each learner a fixed, HARD subset of
columns, decided once at construction and frozen forever. That buys diversity but not
relevance: a learner that happened to draw three noisy columns and one informative one
has no way to lean on the good one more, sample by sample. The predecessor research
document behind this idea frames the fix as a boosting-style update with a learned,
optionally input-dependent SOFT gate multiplying each learner's view of the data:

    F_t(x) = F_{t-1}(x) + eta * h_t(g_t(x) (elementwise*) x)

F_t is the ensemble after t experts, h_t is expert t, and g_t(x) in (0, 1) reweights
the features expert t actually sees before its first Linear layer. eta is already
handled elsewhere in v4 (the per-position step weights in model.py's deep supervision);
this module is only about g_t.

FIVE SWITCHABLE MODES
----------------------
Per the project's rule that no proposed component is assumed to help, gating ships as
five modes, "none" being an exact no-op so it stays the control arm:

  none                  g_t(x) = 1                    identity; today's v4, bit-for-bit.
  global_static         g(x)   = sigmoid(b)            one learned vector over all F input
                                                        columns, shared by every learner,
                                                        input-independent.
  global_dynamic        g(x)   = sigmoid(W x + b)      one small network shared by every
                                                        learner, input-dependent.
  per_learner_static     g_t(x) = sigmoid(b_t)          each learner has its own learned
                                                        vector over its own column subset,
                                                        input-independent.
  per_learner_dynamic    g_t(x) = sigmoid(W_t x_t + b_t) each learner has its own small
                                                        network over its own column subset,
                                                        input-dependent.

"global" vs "per_learner" asks whether relevance is a property of the FEATURE (same for
every expert) or of the (expert, feature) pair (an expert-specific opinion — the thing
feature bagging cannot express). "static" vs "dynamic" asks whether that opinion is fixed
after training or recomputed per sample — the "adaptive" half of the idea.

STARTING AS A NO-OP
--------------------
Every dynamic network's final layer is zero-initialised, so at construction g_t(x) is a
CONSTANT sigmoid(init_bias) regardless of x — dynamic and static modes start identically,
and init_bias defaults to 4.0 (sigmoid(4) ~= 0.982) so every non-"none" mode starts almost
fully open. This matters for a fair ablation: any gap between a gated arm and the ungated
control has to come from what training LEARNS to suppress, not from a disruptive random
gate at epoch 0.

WHERE IT SITS
-------------
The gate reads the RAW (pre-embedding) feature values — gating is "how much of this
column's information should expert t use," which is a property of the column, not of
whatever embeddings.py happened to turn it into. It is applied to whatever tensor the
expert actually consumes (raw scalar or (B, F, d) embedded), broadcasting the per-feature
gate across the embedding dimension. See weak_learners.py's `WeakLearnerEnsemble.forward`
for the wiring.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["FeatureGateConfig", "FeatureGate", "GATE_MODES"]

GATE_MODES = ("none", "global_static", "global_dynamic", "per_learner_static", "per_learner_dynamic")


@dataclass
class FeatureGateConfig:
    """mode       : one of GATE_MODES.
    init_bias  : pre-sigmoid bias every gate starts at (ignored for "none").
                 4.0 -> sigmoid(4.0) ~= 0.982, i.e. "start almost fully open."
    hidden_dim : 0 -> the dynamic gate network is a single Linear(+sigmoid).
                 >0 -> Linear -> ReLU -> Linear, hidden width hidden_dim.
                 Ignored for "none" and the two static modes.
    """

    mode: str = "none"
    init_bias: float = 4.0
    hidden_dim: int = 0

    def __post_init__(self):
        if self.mode not in GATE_MODES:
            raise ValueError(f"mode must be one of {GATE_MODES}, got {self.mode!r}")
        if self.hidden_dim < 0:
            raise ValueError(f"hidden_dim must be >= 0, got {self.hidden_dim}")


def _make_gate_net(in_features: int, out_features: int, hidden_dim: int, init_bias: float) -> nn.Module:
    """A tiny (optionally 1-hidden-layer) net whose OUTPUT starts as a constant.

    Zeroing the final layer's weight makes the network's output equal to its bias alone,
    for any input, at construction time — a dynamic gate that has not yet learned to
    react to x is indistinguishable from a static one. Training is what breaks the tie.
    """
    if hidden_dim > 0:
        net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_features))
        final = net[-1]
    else:
        net = nn.Linear(in_features, out_features)
        final = net
    nn.init.zeros_(final.weight)
    nn.init.constant_(final.bias, init_bias)
    return net


class FeatureGate(nn.Module):
    """Computes g_t(x) for every weak learner at once.

    forward(x_raw) -> None (mode "none": no gating, callers must skip multiplying) or a
    (B, K, k) tensor of values in (0, 1), one row per learner, columns aligned to that
    learner's OWN feat_idx order (k = features per learner — feature bagging always draws
    the same subset size for every learner, so this is well-defined).
    """

    def __init__(self, in_dim: int, feature_subsets: Sequence[np.ndarray],
                 config: Optional[FeatureGateConfig] = None, **overrides):
        super().__init__()
        if config is None:
            config = FeatureGateConfig(**overrides)
        elif overrides:
            config = FeatureGateConfig(**{**config.__dict__, **overrides})
        self.config = config
        self.in_dim = in_dim
        self.num_learners = len(feature_subsets)

        sizes = {len(s) for s in feature_subsets}
        if len(sizes) != 1:
            raise ValueError(
                "FeatureGate assumes every learner's feature subset has the same size "
                f"(feature bagging always produces this); got sizes {sorted(sizes)}"
            )
        self.k = sizes.pop()

        idx = np.stack([np.asarray(s, dtype=np.int64) for s in feature_subsets])   # (K, k)
        self.register_buffer("feat_idx", torch.from_numpy(idx), persistent=True)

        mode = config.mode
        self.raw_gate: Optional[nn.Parameter] = None
        self.net: Optional[nn.Module] = None

        if mode == "none":
            pass
        elif mode == "global_static":
            self.raw_gate = nn.Parameter(torch.full((in_dim,), config.init_bias))
        elif mode == "per_learner_static":
            self.raw_gate = nn.Parameter(torch.full((self.num_learners, self.k), config.init_bias))
        elif mode == "global_dynamic":
            self.net = _make_gate_net(in_dim, in_dim, config.hidden_dim, config.init_bias)
        elif mode == "per_learner_dynamic":
            self.net = nn.ModuleList(
                _make_gate_net(self.k, self.k, config.hidden_dim, config.init_bias)
                for _ in range(self.num_learners)
            )

    def forward(self, x_raw: torch.Tensor) -> Optional[torch.Tensor]:
        mode = self.config.mode
        if mode == "none":
            return None
        if x_raw.dim() != 2 or x_raw.size(1) != self.in_dim:
            raise ValueError(f"expected x_raw of shape (B, {self.in_dim}), got {tuple(x_raw.shape)}")
        B = x_raw.shape[0]

        if mode == "global_static":
            logits = self.raw_gate[self.feat_idx]                              # (K, k)
            return torch.sigmoid(logits).unsqueeze(0).expand(B, -1, -1)        # (B, K, k)
        if mode == "per_learner_static":
            return torch.sigmoid(self.raw_gate).unsqueeze(0).expand(B, -1, -1)

        if mode == "global_dynamic":
            logits_full = self.net(x_raw)                                     # (B, in_dim)
            logits = logits_full[:, self.feat_idx]                            # (B, K, k)
            return torch.sigmoid(logits)

        # per_learner_dynamic: each learner's own tiny net sees only its own columns.
        xi = x_raw[:, self.feat_idx]                                          # (B, K, k)
        logits = torch.stack([net(xi[:, i, :]) for i, net in enumerate(self.net)], dim=1)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"mode={self.config.mode}, K={self.num_learners}, k={self.k}, in_dim={self.in_dim}"


# ---------------------------------------------------------------------------
# Smoke test:  python feature_gating.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, IN_DIM, K, FRAC = 64, 10, 6, 0.7

    rng = np.random.RandomState(0)
    k = int(round(FRAC * IN_DIM))
    subsets = [np.sort(rng.choice(IN_DIM, size=k, replace=False)) for _ in range(K)]
    x = torch.randn(B, IN_DIM)

    print("SHAPES AND SIZES")
    gates = {}
    for mode in GATE_MODES:
        g = FeatureGate(IN_DIM, subsets, FeatureGateConfig(mode=mode))
        out = g(x)
        shape = tuple(out.shape) if out is not None else None
        print(f"  {mode:20s} -> {str(shape):16s} params {g.num_parameters():,}")
        gates[mode] = g

    print("\nCHECKS")
    ok = gates["none"](x) is None
    print(f"  [{'PASS' if ok else 'FAIL'}] mode='none' returns None (no gating happens at all)")

    ok = gates["global_static"].num_parameters() == IN_DIM
    print(f"  [{'PASS' if ok else 'FAIL'}] global_static has exactly in_dim={IN_DIM} parameters (one per column)")

    ok = gates["per_learner_static"].num_parameters() == K * k
    print(f"  [{'PASS' if ok else 'FAIL'}] per_learner_static has K*k={K*k} parameters (independent per learner)")

    # Every non-"none" mode starts near sigmoid(init_bias) ~= 0.982, i.e. "almost fully open".
    expected_open = torch.sigmoid(torch.tensor(4.0)).item()
    all_near_open = all(
        torch.allclose(gates[m](x), torch.full_like(gates[m](x), expected_open), atol=1e-3)
        for m in ("global_static", "per_learner_static")
    )
    print(f"  [{'PASS' if all_near_open else 'FAIL'}] static modes start at sigmoid(init_bias)={expected_open:.3f} "
          f"('almost fully open')")

    # Dynamic modes: zero-init final layer means output is CONSTANT across different x at init.
    x2 = torch.randn(B, IN_DIM)
    for mode in ("global_dynamic", "per_learner_dynamic"):
        a, b = gates[mode](x), gates[mode](x2)
        ok = torch.allclose(a, b, atol=1e-5) and torch.allclose(a, torch.full_like(a, expected_open), atol=1e-3)
        print(f"  [{'PASS' if ok else 'FAIL'}] {mode} is a CONSTANT at init (zero-init trick worked: "
              f"same output for two different inputs, both ~= {expected_open:.3f})")

    # After a training step, dynamic modes SHOULD start reacting to x; static modes never will.
    print("\nAFTER ONE GRADIENT STEP (does the gate learn to react to the input?)")
    for mode in ("global_static", "global_dynamic", "per_learner_static", "per_learner_dynamic"):
        g = FeatureGate(IN_DIM, subsets, FeatureGateConfig(mode=mode))
        opt = torch.optim.SGD(g.parameters(), lr=1.0)
        target = x.mean(dim=1)   # something a gate could only fit by reacting to x
        for _ in range(20):
            out = g(x)
            loss = (out.mean(dim=(1, 2)) - target).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a, b = g(x), g(x2)
        reacts = not torch.allclose(a, b, atol=1e-4)
        is_dynamic = "dynamic" in mode
        ok = reacts == is_dynamic
        print(f"  [{'PASS' if ok else 'FAIL'}] {mode:20s} reacts to different inputs after training: {reacts}"
              f"  (expected {is_dynamic})")

    # Gradient actually reaches every parameter (nothing is silently detached).
    print("\nGRADIENT FLOW")
    for mode in GATE_MODES[1:]:
        g = FeatureGate(IN_DIM, subsets, FeatureGateConfig(mode=mode))
        out = g(x)
        out.sum().backward()
        grads = [p.grad for p in g.parameters() if p.requires_grad]
        ok = len(grads) > 0 and all(gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0 for gr in grads)
        print(f"  [{'PASS' if ok else 'FAIL'}] {mode:20s} every parameter receives a finite, nonzero gradient")
