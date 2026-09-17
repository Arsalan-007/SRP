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
    Sequential Residual Predictor (v2):
      1. Weak learner ensemble -> (B, K, D)
      2. Add learnable learner-position embeddings
      3. Residual causal Transformer blocks
      4. Each position predicts -> (B, K, output_dim)
    
    NO CLS token - each learner makes a prediction after seeing previous learners.
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
        # self.head = nn.Linear(embed_dim, output_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(), # SiLU/Swish handles continuous non-linear data beautifully
            nn.Linear(embed_dim, output_dim)
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self._init_head()

    # def _init_head(self):
    #     nn.init.xavier_uniform_(self.head.weight)
    #     nn.init.zeros_(self.head.bias)

    def _init_head(self):
        """
        Iterate through the layers of the head and initialize 
        the Linear layers properly.
        """
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # def forward(self, x):
    #     features = self.ensemble(x)
    #     # h = self.input_norm(features)
    #     # h = self.dropout(h + self.pos_embedding[:, :h.size(1)])
    #     attn_output, attn_weights = self.attention(features)
    #     norm_output = self.norm(features + attn_output)
    #     # attn_weights = None
    #     predictions = self.head(norm_output)
    #     # for block in self.blocks:
    #     #     h, attn_weights = block(h)

    #     # h = self.final_norm(h)
    #     # predictions = self.head(h)

    #     if self.task == "regression":
    #         predictions = predictions.squeeze(-1)

    #     return predictions, features, attn_weights
    def forward(self, x):
        # 1. Get features from all K weak learners
        features = self.ensemble(x) # Shape: (Batch, NUM_LEARNERS, Embed_Dim)

        # 2. Add Positional Embeddings and Initial Norm
        h = self.input_norm(features)
        h = self.dropout(h + self.pos_embedding[:, :h.size(1)])

        # 3. Pass through the Causal Transformer Blocks
        attn_weights_list = []
        for block in self.blocks:
            h, attn_weights = block(h)
            attn_weights_list.append(attn_weights)
            
        # Optional: You can choose to return the attention weights from the last block, 
        # or all of them depending on your visualization needs.
        final_attn_weights = attn_weights_list[-1] 

        # 4. Final Norm
        norm_output = self.final_norm(h)

        # 5. Pass ALL steps to the prediction head
        predictions = self.head(norm_output)

        # 6. Squeeze the last dimension if it's a regression task so shape is (Batch, K)
        if self.task == 'regression' and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        return predictions, features, final_attn_weights
    
    def get_feature_importance(self, x):
        """Return attention-based feature importance for the last learner."""
        _, features, attn_weights = self.forward(x)
        if attn_weights is not None:
            # Average attention weights across all heads for simplicity
            attn_weights = attn_weights.mean(dim=1)  # (B, K, K)
            # Importance is the sum of attention weights for each feature
            importance = attn_weights[:, -1, :].detach()  # Last learner's attention
            return importance
        else:
            raise ValueError("Attention weights not computed")
    


    def predict(self, x, weight_scheme='exponential', **weight_kwargs):
        """
        Dedicated inference method. 
        Automatically applies eval mode, disables gradients, 
        and returns a weighted average of all predictions.
        
        Args:
            x: Input tensor
            weight_scheme: Weighting scheme - 'linear', 'uniform', or 'exponential'
            **weight_kwargs: Additional arguments for WeightScheme (e.g., base for exponential)
        """
        from weight_schemes import WeightScheme
        
        # 1. Store the current mode so we can restore it later
        was_training = self.training
        self.eval()
        
        # 2. Run forward pass without tracking gradients
        with torch.no_grad():
            predictions, _, _ = self.forward(x)
            
            # 3. Compute weighted average using the specified scheme
            # Shape: (Batch, K) - K predictions per sample
            K = predictions.shape[1]
            device = predictions.device
            
            # Get weights using WeightScheme
            ws = WeightScheme(method=weight_scheme, **weight_kwargs)
            weights = ws.get_weights(K, device)
            
            # Weighted average: sum(preds * weights) / sum(weights)
            # Since weights are normalized, this is just sum(preds * weights)
            final_prediction = (predictions * weights).sum(dim=-1)
            
        # 4. Restore the model to its previous state
        if was_training:
            self.train()
            
        return final_prediction