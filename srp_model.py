import torch
import torch.nn as nn


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal self-attention block over weak learner tokens."""
    def __init__(self, embed_dim, num_heads=2, dropout=0.1, ff_mult=4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        ff_dim = max(embed_dim * ff_mult, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


class SRPModel(nn.Module):
    """
    Sequential Residual Predictor:
      1. Weak learner ensemble -> (B, K, D)
      2. Add learnable learner-position embeddings
      3. Residual causal Transformer blocks
      4. Deep supervision: every step predicts, final step sees all previous learners
    """
    def __init__(self, input_dim, num_learners=8, hidden_dim=8,
                 embed_dim=16, depth=1, num_heads=2,
                 output_dim=1, task="regression", attn_depth=2,
                 dropout=0.1, ff_mult=4):
        super().__init__()
        from weak_learners import WeakLearnerEnsemble

        self.task = task
        self.num_learners = num_learners
        self.output_dim = output_dim

        self.ensemble = WeakLearnerEnsemble(
            input_dim, num_learners, hidden_dim, embed_dim, depth
        )

        self.input_norm = nn.LayerNorm(embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_learners, embed_dim))
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, dropout, ff_mult)
            for _ in range(attn_depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, output_dim)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self._init_head()

    def _init_head(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.ensemble(x)
        h = self.input_norm(features)
        h = self.dropout(h + self.pos_embedding[:, :h.size(1)])

        attn_weights = None
        for block in self.blocks:
            h, attn_weights = block(h)

        h = self.final_norm(h)
        predictions = self.head(h)

        if self.task == "regression":
            predictions = predictions.squeeze(-1)

        return predictions, features, attn_weights

    def predict_final(self, x):
        """Return only the final sequential prediction for evaluation/inference."""
        predictions, features, attn_weights = self(x)
        final_prediction = predictions[:, -1]
        return final_prediction, features, attn_weights
