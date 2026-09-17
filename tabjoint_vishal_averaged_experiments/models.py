
import math
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- Feature subsets ----------
@dataclass
class FeatureSubspaceCfg:
    frac: float = 0.6
    resample_each_epoch: bool = False
    seed: int = 42

def make_feature_subsets(in_dim: int, n_mlps: int, frac: float, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    k = max(1, int(np.ceil(frac * in_dim)))
    all_idx = np.arange(in_dim)
    subsets = []
    for i in range(n_mlps):
        idx = rng.choice(all_idx, size=k, replace=False)
        subsets.append(np.sort(idx))
    return subsets

# ---------- Backbones & transformer ----------
class ShallowBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_hidden_layers: int, dropout: float):
        super().__init__()
        layers=[]; dim=in_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.emb_dim = dim
    def forward(self, x): return self.backbone(x)

@dataclass
class TCfg:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1

class EmbeddingsTransformer(nn.Module):
    def __init__(self, emb_dim: int, n_mlps: int, out_dim: int, task: str, cfg: TCfg):
        super().__init__()
        self.task, self.n_mlps = task, n_mlps
        self.in_proj = nn.Linear(emb_dim, cfg.d_model) if emb_dim != cfg.d_model else nn.Identity()
        self.cls = nn.Parameter(torch.zeros(1,1,cfg.d_model)); nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos_emb = nn.Embedding(n_mlps+1, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, out_dim)
    def forward(self, tokens):  # (B, n_mlps, emb_dim)
        B,T,_ = tokens.shape; assert T==self.n_mlps
        h = self.in_proj(tokens)                  # (B,T,d_model)
        cls = self.cls.expand(B,1,-1)             # (B,1,d_model)
        h = torch.cat([cls,h], dim=1)             # (B,T+1,d_model)
        pos_ids = torch.arange(T+1, device=h.device).unsqueeze(0).expand(B,-1)
        h = h + self.pos_emb(pos_ids)
        z = self.encoder(h)                       # (B,T+1,d_model)
        cls_out = z[:,0,:]                        # (B,d_model)
        logits = self.head(cls_out)               # (B,out_dim)
        return logits, cls_out

# ---------- Full joint model ----------
class JointMLPTransformer(nn.Module):
    def __init__(self, in_dim: int, n_mlps: int, emb_dim: int,
                 mlp_hidden_layers: int, mlp_dropout: float,
                 out_dim: int, task: str, tcfg: TCfg,
                 subspace_cfg: Optional[FeatureSubspaceCfg] = None):
        super().__init__()
        self.task = task
        self.n_mlps = n_mlps
        self.in_dim = in_dim

        if subspace_cfg is None:
            subspace_cfg = FeatureSubspaceCfg(frac=0.6, resample_each_epoch=False, seed=42)
        self.subspace_cfg = subspace_cfg
        self.feature_subsets = make_feature_subsets(in_dim, n_mlps, subspace_cfg.frac, subspace_cfg.seed)

        self.mlps = nn.ModuleList([
            ShallowBackbone(in_dim=len(self.feature_subsets[i]),
                            hidden=emb_dim,
                            n_hidden_layers=mlp_hidden_layers,
                            dropout=mlp_dropout)
            for i in range(n_mlps)
        ])
        assert all(m.emb_dim == emb_dim for m in self.mlps), "emb_dim mismatch"

        self.agg = EmbeddingsTransformer(emb_dim=emb_dim, n_mlps=n_mlps, out_dim=out_dim, task=task, cfg=tcfg)

    @torch.no_grad()
    def resample_feature_subsets(self):
        self.feature_subsets = make_feature_subsets(self.in_dim, self.n_mlps, self.subspace_cfg.frac, self.subspace_cfg.seed)

    def forward(self, x):  # x: (B, in_dim)
        embs = []
        for i, mlp in enumerate(self.mlps):
            cols = self.feature_subsets[i]
            xi = x[:, cols]                     # (B, k_i)
            ei = mlp(xi)                        # (B, emb_dim)
            embs.append(ei)
        tokens = torch.stack(embs, dim=1)       # (B, n_mlps, emb_dim)
        logits, cls_out = self.agg(tokens)      # (B, out_dim), (B, d_model)
        return logits, tokens, cls_out

# ---------- Loss/metric ----------
def make_loss(task: str):
    if task=="binary": return nn.BCEWithLogitsLoss()
    if task=="multiclass": return nn.CrossEntropyLoss()
    return nn.MSELoss()

def metric(task: str, logits, y) -> float:
    if task=="binary":
        preds = (torch.sigmoid(logits.view(-1))>0.5).long(); return (preds==y.long()).float().mean().item()
    if task=="multiclass":
        preds = logits.argmax(dim=1); return (preds==y.long()).float().mean().item()
    mse = F.mse_loss(logits.view_as(y).float(), y.float()).item(); return -math.sqrt(mse)

def diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    B,n,d = tokens.shape
    e = torch.nn.functional.normalize(tokens, dim=2)
    sim = e @ e.transpose(1,2)
    eye = torch.eye(n, device=tokens.device).unsqueeze(0)
    return (sim*(1-eye)).mean()

@dataclass
class TrainCfg:
    epochs:int=30; batch_size:int=512
    lr_mlp:float=5e-4; lr_tr:float=1e-3
    weight_decay:float=1e-5; patience:int=6
    lambda_div:float=0.0
    lambda_cka:float=0.035

def cka_offdiag_mean(tokens: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean pairwise linear CKA across MLP tokens (off-diagonal)."""
    B, n, d = tokens.shape
    Z = tokens - tokens.mean(dim=0, keepdim=True)          # (B, n, d)
    norms = []
    Z_list = []
    for i in range(n):
        Xi = Z[:, i, :]
        Z_list.append(Xi)
        XiTXi = Xi.T @ Xi
        norms.append(torch.linalg.norm(XiTXi, ord='fro') + eps)
    norms = torch.stack(norms)
    cka_sum = tokens.new_zeros(())
    pairs = 0
    for i in range(n):
        Xi = Z_list[i]
        for j in range(i + 1, n):
            Yj = Z_list[j]
            XiTYj = Xi.T @ Yj
            num = (XiTYj ** 2).sum()
            cka = (num / (norms[i] * norms[j])).clamp(0.0, 1.0)
            cka_sum = cka_sum + cka
            pairs += 1
    return (cka_sum / max(pairs, 1))
