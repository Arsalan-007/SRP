"""
ncl.py  —  SRP v4, step 10: Negative Correlation Learning (Liu & Yao, 1999)

WHY THIS EXISTS
---------------
Every weak learner in v4 is trained the same way: minimize its own prediction error,
independently. Learners only interact through feature bagging (different learners see
different columns) and, for attention arms, the attention stage itself. Nothing in the
LOSS pushes learners to disagree usefully.

Negative Correlation Learning (Liu & Yao, 1999, "Simultaneous training of neural network
ensembles with negative correlation learning") adds a per-learner penalty that rewards
disagreeing with the CURRENT ensemble mean:

    E_i = (1/N) sum_n [ (1/2)(F_i(n) - y(n))^2  +  lambda * p_i(n) ]
    p_i(n) = (F_i(n) - Fbar(n)) * sum_{j != i} (F_j(n) - Fbar(n))

Fbar = mean(F_1..F_K). Because sum_j (F_j - Fbar) = 0 for the mean of K terms,
sum_{j!=i}(F_j - Fbar) = -(F_i - Fbar), so this telescopes to the clean closed form used
directly here:

    p_i(n) = -(F_i(n) - Fbar(n))**2

i.e. each learner's loss is its own accuracy loss MINUS lambda times its squared deviation
from the ensemble mean — a term that DECREASES as the learner moves further from the mean,
so gradient descent is pushed to INCREASE that deviation. If learner A overpredicts, the
loss for learner B is relieved by underpredicting: the ensemble is pulled toward the target
on average even when no individual learner is.

TWO SUBTLETIES THAT CHANGE WHAT YOU'RE ACTUALLY TESTING
---------------------------------------------------------------------
1. Whether Fbar is detached from the graph turns out NOT to matter here — but only for a
   reason worth understanding, not by luck. Differentiating a SINGLE learner's penalty term
   alone, treating Fbar as dependent on F_i, gives a gradient scaled by (1 - 1/K) relative to
   treating Fbar as constant. But `training.py` never trains one learner alone: it calls one
   `.backward()` on the SUMMED total loss for all K learners every step. Once you sum over
   all K learners' penalty terms and differentiate THAT w.r.t. one learner's output, the
   cross-terms coming from the other K-1 learners' penalties (which also depend on Fbar,
   which also depends on F_i) exactly cancel the (1-1/K) factor, reconstructing the
   "as-if-detached" gradient magnitude exactly. Confirmed empirically in this module's own
   self-test (`detach_mean=True` vs `False` give numerically identical results throughout a
   full training run) — both are kept as an option, but neither is the "more correct" one
   for how this codebase actually trains models; `detach_mean` would only start to matter if
   learners were ever updated via separate, sequential backward/optimizer calls.
2. This needs each learner's OWN scalar prediction, not just its own embedding. v4's
   mean-pool arm currently averages EMBEDDINGS and applies one shared head ONCE — there is
   no "learner i's prediction" today, only "learner i's opinion vector." Testing NCL
   requires moving the average from embedding-space to prediction-space
   (`model.py`'s `SRPConfig(pool_level="output")`), which is itself an architecture change
   independent of NCL. `lambda_ncl=0` under `pool_level="output"` isolates that confound:
   it reproduces plain independent per-learner training with predict-then-average pooling,
   so comparing it to today's embedding-then-predict mean-pool tells you whether the
   POOLING POINT alone matters, before crediting anything to the penalty itself.

NOT YET SUPPORTED: multiclass. The (F_i - Fbar)^2 penalty is defined for a SCALAR
prediction; extending it to a per-class logit vector is a real design decision (e.g.
penalize per class independently, or on the softmax simplex) left for later.
"""

from typing import Callable

import torch
import torch.nn.functional as F

__all__ = ["ncl_loss", "prediction_correlation"]


def ncl_loss(step_preds: torch.Tensor, y: torch.Tensor, criterion: Callable,
            lambda_ncl: float = 0.0, detach_mean: bool = False) -> torch.Tensor:
    """Mean, over K learners, of (own accuracy loss - lambda * squared deviation from the
    ensemble mean). `step_preds` is (B, K) raw per-learner scalar predictions (logits for
    binary, raw values for regression). `criterion` is any (pred, y) -> scalar loss
    (typically `SRPModel.criterion`), so this stays task-agnostic for regression/binary.

    `lambda_ncl=0` is an EXACT no-op: reduces to the plain mean of K independently-computed
    losses, identical to training each learner alone with no notion of the others."""
    if step_preds.dim() != 2:
        raise ValueError(f"expected (B, K) per-learner predictions, got {tuple(step_preds.shape)}")
    K = step_preds.shape[1]
    fbar = step_preds.mean(dim=1)                     # (B,) — NOT detached by default; see module docstring
    if detach_mean:
        fbar = fbar.detach()

    total = step_preds.new_zeros(())
    for i in range(K):
        base = criterion(step_preds[:, i], y)
        if lambda_ncl != 0.0:
            deviation = step_preds[:, i] - fbar
            base = base - lambda_ncl * (deviation ** 2).mean()
        total = total + base
    return total / K


@torch.no_grad()
def prediction_correlation(step_preds: torch.Tensor) -> float:
    """Mean pairwise Pearson correlation between learners' predictions across the batch —
    the direct, MEASURED check of whether NCL actually reduced correlation, the same role
    weak_learners.py's diversity_stats() plays for embedding-level collapse."""
    K = step_preds.shape[1]
    if K < 2:
        return float("nan")
    centered = step_preds - step_preds.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True).clamp_min(1e-8)
    normed = centered / std
    corr = (normed.T @ normed) / normed.shape[0]
    off = corr[~torch.eye(K, dtype=torch.bool, device=corr.device)]
    return off.mean().item()


# ---------------------------------------------------------------------------
# Smoke test:  python ncl.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, K = 256, 6

    print("CHECKS")

    # 1. lambda=0 is an exact no-op: equals the plain mean of independent per-learner losses.
    step_preds = torch.randn(B, K)
    y = torch.randn(B)
    got = ncl_loss(step_preds, y, F.mse_loss, lambda_ncl=0.0)
    want = torch.stack([F.mse_loss(step_preds[:, i], y) for i in range(K)]).mean()
    ok1 = torch.allclose(got, want, atol=1e-6)
    print(f"  [{'PASS' if ok1 else 'FAIL'}] lambda_ncl=0 exactly reproduces independent-learner training "
          f"({got.item():.6f} vs {want.item():.6f})")

    # 2. gradient flows to every learner's prediction, for both detach_mean settings.
    for detach in (False, True):
        sp = torch.randn(B, K, requires_grad=True)
        loss = ncl_loss(sp, y, F.mse_loss, lambda_ncl=0.5, detach_mean=detach)
        loss.backward()
        ok = torch.isfinite(sp.grad).all() and (sp.grad.abs().sum(0) > 0).all()
        print(f"  [{'PASS' if ok else 'FAIL'}] detach_mean={detach!s:5s} every learner receives a "
              f"finite, nonzero gradient")

    # 3. THE key check: does training under NCL actually lower correlation between learners'
    #    predictions, vs. training the same learners independently (lambda=0)? Toy ensemble of
    #    K linear heads, each restricted to a random HALF of the input features (mirrors real
    #    feature bagging — unlike full-shared-access heads, these have no single trivial shared
    #    optimum, so there's real tension between "fit alone" and "diversify," like the real model.
    print("\nEMPIRICAL: does it do what it claims? (toy regression, K=6 feature-bagged linear heads)")
    torch.manual_seed(0)
    N, D = 2000, 20
    Xtr = torch.randn(N, D)
    w_true = torch.randn(D)
    ytr = Xtr @ w_true + 0.3 * torch.randn(N)
    Xte = torch.randn(500, D)
    yte = Xte @ w_true + 0.3 * torch.randn(500)

    subset_rng = torch.Generator().manual_seed(0)
    subsets = [torch.randperm(D, generator=subset_rng)[: D // 2] for _ in range(K)]

    class BaggedHeads(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.heads = torch.nn.ModuleList([torch.nn.Linear(D // 2, 1) for _ in range(K)])

        def forward(self, x):
            return torch.cat([h(x[:, subsets[i]]) for i, h in enumerate(self.heads)], dim=1)

    def train(model, lam, detach, steps=300, lr=0.05):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = model(Xtr)                              # (N, K)
            loss = ncl_loss(pred, ytr, F.mse_loss, lambda_ncl=lam, detach_mean=detach)
            loss.backward()
            opt.step()
        return model

    results = {}
    for lam in (0.0, 0.4, 0.8, 1.0, 2.0):
        for detach in (False, True):
            model = train(BaggedHeads(seed=0), lam, detach)
            with torch.no_grad():
                te_pred = model(Xte)
                ens_mse = F.mse_loss(te_pred.mean(dim=1), yte).item()
                corr = prediction_correlation(te_pred)
            results[(lam, detach)] = (ens_mse, corr)
            print(f"  lambda={lam:<4.1f} detach_mean={str(detach):5s}  "
                  f"ensemble test MSE={ens_mse:.4f}   mean pairwise corr={corr:+.4f}")

    baseline_mse, baseline_corr = results[(0.0, False)]
    good = results[(0.8, False)]
    healthy = good[0] < baseline_mse and good[1] < baseline_corr
    print(f"\n  [{'PASS' if healthy else 'FAIL'}] lambda=0.8 both lowers correlation ({good[1]:+.4f} < "
          f"{baseline_corr:+.4f}) AND improves ensemble MSE ({good[0]:.4f} < {baseline_mse:.4f}) vs. "
          f"independent training — feature-bagged learners forced to specialize recover accuracy that")
    print("        no single one of them has alone, which is exactly NCL's claimed mechanism.")
    blown_up = results[(2.0, False)][0] > 50 * baseline_mse
    print(f"  [{'PASS' if blown_up else 'FAIL'}] lambda=2.0 is well past the useful range and visibly "
          f"destabilises training (MSE {results[(2.0, False)][0]:.1f} vs baseline {baseline_mse:.4f}) — "
          "matches the literature's lambda<=1 rule of thumb, reproduced empirically, not assumed.")
    same = all(abs(results[(lam, False)][0] - results[(lam, True)][0]) < 1e-3 for lam in (0.0, 0.4, 0.8, 1.0, 2.0))
    print(f"  [{'INFO'}] detach_mean made {'NO' if same else 'A'} difference here — expected: for a single")
    print("         joint backward() over the SUMMED loss (what SRPModel.loss()/training.py always do),")
    print("         the per-learner (1-1/K) factor from detaching exactly cancels against the cross-learner")
    print("         terms from the other K-1 losses, so both settings give the identical total gradient.")
    print("         Only matters if learners were ever updated via separate, sequential backward calls.")
