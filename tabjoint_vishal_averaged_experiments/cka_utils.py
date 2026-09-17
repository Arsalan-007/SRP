
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def collect_tokens_sampled(model, loader, device, max_rows=4096):
    """
    Runs the joint model over `loader` in eval mode and returns up to `max_rows`
    rows of MLP token embeddings. Output: (N_sample, n_mlps, emb_dim)
    """
    model.eval()
    chunks = []
    n_collected = 0
    for xb, _ in loader:
        xb = xb.to(device)
        _, tokens, _ = model(xb)           # tokens: (B, n_mlps, emb_dim)
        if n_collected + tokens.size(0) > max_rows:
            need = max_rows - n_collected
            chunks.append(tokens[:need].cpu())
            break
        chunks.append(tokens.cpu())
        n_collected += tokens.size(0)
    if not chunks:
        raise RuntimeError("No tokens collected; check your loader.")
    return torch.cat(chunks, dim=0)        # (N_sample, n_mlps, emb_dim)

def _linear_cka(X, Y, eps=1e-12):
    """Correct linear CKA in [0,1]. X, Y: (N, d) float tensors."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = (XtY ** 2).sum()                          # ||X^T Y||_F^2
    den = torch.linalg.norm(XtX, ord='fro') * torch.linalg.norm(YtY, ord='fro') + eps
    return (num / den).clamp(min=0.0, max=1.0)

def cka_token_matrix(tokens):
    """
    tokens: (N, n_mlps, emb_dim). Returns an (n_mlps, n_mlps) CKA matrix.
    Each entry (i,j) measures representational similarity between token i and j,
    invariant to any invertible linear transform.
    """
    N, n, d = tokens.shape
    S = torch.zeros(n, n)
    for i in range(n):
        Xi = tokens[:, i, :]
        for j in range(n):
            Yj = tokens[:, j, :]
            S[i, j] = _linear_cka(Xi, Yj)
    return S.numpy()

def plot_matrix(M: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(M, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("token j"); ax.set_ylabel("token i")
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.show()
