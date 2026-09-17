
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt

from .stats_utils import offdiag_stats  # shared

@torch.no_grad()
def avg_similarity_matrix_over_loader(model, loader, device) -> np.ndarray:
    """
    For JointMLPTransformer: run over a loader and return the
    mean cosine-similarity matrix across the batch dimension.
    Output: (n_mlps, n_mlps)
    """
    model.eval()
    n_mlps = model.agg.n_mlps
    sim_sum = torch.zeros(n_mlps, n_mlps, device=device)
    count = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, tokens, _ = model(xb)           # tokens: (B, n_mlps, emb_dim)
            e = F.normalize(tokens, dim=2)     # L2-normalize per token so dot=cosine
            sim = e @ e.transpose(1, 2)        # (B, n_mlps, n_mlps)
            sim_sum += sim.mean(dim=0)         # average over the batch
            count += 1
    return (sim_sum / max(count, 1)).detach().cpu().numpy()

def plot_sim_matrix(S: np.ndarray, title="Avg cosine similarity (MLP tokens)"):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(S, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("token j")
    ax.set_ylabel("token i")
    n = S.shape[0]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()
