import torch
import torch.nn.functional as F


def cosine_similarity_matrix(features):
    """
    features: (B, K, D)
    Returns mean of off-diagonal cosine similarities averaged over batch.
    We MAXIMIZE diversity => MINIMIZE this value.
    """
    # Average over batch first
    avg = features.mean(dim=0)                          # (K, D)
    normed = F.normalize(avg, dim=-1)                   # (K, D)
    sim_matrix = normed @ normed.T                      # (K, K)

    K = sim_matrix.size(0)
    mask = ~torch.eye(K, dtype=torch.bool, device=features.device)
    off_diag = sim_matrix[mask]                         # K*(K-1) values
    return off_diag.mean(), sim_matrix


def pearson_correlation_matrix(features):
    """
    features: (B, K, D)
    Computes Pearson correlation between each pair of learner output vectors
    averaged over the batch. Returns mean off-diagonal correlation.
    """
    avg = features.mean(dim=0)                          # (K, D)
    avg = avg - avg.mean(dim=-1, keepdim=True)          # center
    std = avg.std(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = avg / std                                  # (K, D)
    corr_matrix = (normed @ normed.T) / avg.size(-1)   # (K, K)

    K = corr_matrix.size(0)
    mask = ~torch.eye(K, dtype=torch.bool, device=features.device)
    off_diag = corr_matrix[mask].abs()
    return off_diag.mean(), corr_matrix


def diversity_loss(features, mode="cosine"):
    """
    features: (B, K, D)
    mode: "cosine" | "correlation" | "combined"
    Returns scalar loss to minimize (lower = more diverse).
    """
    if mode == "cosine":
        loss, matrix = cosine_similarity_matrix(features)
    elif mode == "correlation":
        loss, matrix = pearson_correlation_matrix(features)
    elif mode == "combined":
        cos_loss, cos_mat = cosine_similarity_matrix(features)
        cor_loss, cor_mat = pearson_correlation_matrix(features)
        loss = 0.5 * cos_loss + 0.5 * cor_loss
        matrix = (cos_mat + cor_mat) / 2
    else:
        raise ValueError(f"Unknown diversity mode: {mode}")
    return loss, matrix
