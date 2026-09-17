import torch
import torch.nn as nn


class WeakLearnerMLP(nn.Module):
    """
    A small, constrained MLP (weak learner).
    depth: 1 or 2 hidden layers
    width: 4-16 neurons
    """
    def __init__(self, input_dim, hidden_dim=8, output_dim=16, depth=1):
        super().__init__()
        assert depth in (1, 2), "depth must be 1 or 2"
        assert 4 <= hidden_dim <= 16, "width should be between 4 and 16"

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if depth == 2:
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, output_dim)


class WeakLearnerEnsemble(nn.Module):
    """
    Ensemble of K weak learner MLPs.
    Returns stacked feature vectors: (B, K, output_dim)
    """
    def __init__(self, input_dim, num_learners=8, hidden_dim=8, output_dim=16, depth=1):
        super().__init__()
        self.learners = nn.ModuleList([
            WeakLearnerMLP(input_dim, hidden_dim, output_dim, depth)
            for _ in range(num_learners)
        ])

    def forward(self, x):
        # x: (B, input_dim)
        outputs = [mlp(x) for mlp in self.learners]  # K x (B, output_dim)
        return torch.stack(outputs, dim=1)            # (B, K, output_dim)

    def l1_penalty(self, normalize=True, include_bias=False):
        """Compute L1 norm over weak learner weights.

        By default, this excludes bias terms and normalizes by the
        total number of weight parameters so the penalty scale stays
        stable across model sizes.
        """
        l1 = 0.0
        total_params = 0
        for mlp in self.learners:
            for name, p in mlp.named_parameters():
                if p.requires_grad and (include_bias or p.dim() > 1):
                    l1 = l1 + p.abs().sum()
                    total_params += p.numel()
        return l1 / max(1, total_params) if normalize else l1

    def l1_l2_penalty(self, l1_lambda=1e-4, l2_lambda=1e-4, include_bias=False):
        """Compute combined L1+L2 weight penalty across all weak learners."""
        l1 = 0.0
        l2 = 0.0
        for mlp in self.learners:
            for name, p in mlp.named_parameters():
                if p.requires_grad and (include_bias or p.dim() > 1):
                    l1 = l1 + p.abs().sum()
                    l2 = l2 + p.pow(2).sum()
        return l1_lambda * l1 + l2_lambda * l2
