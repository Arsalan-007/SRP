
import numpy as np, torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import ArrayDataset
from .models import JointMLPTransformer, FeatureSubspaceCfg, make_loss, metric, cka_offdiag_mean

def train_joint(X_train_t, y_train, X_val_t, y_val, X_test_t, y_test,
                in_dim, n_mlps, emb_dim, mlp_hidden_layers, mlp_dropout,
                out_dim, task, tcfg, cfg, device):
    ds_tr = ArrayDataset(X_train_t, y_train)
    ds_va = ArrayDataset(X_val_t,   y_val)
    ds_te = ArrayDataset(X_test_t,  y_test)
    tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)
    te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False)
    sub_cfg = FeatureSubspaceCfg(frac=0.6, resample_each_epoch=True, seed=123)
    model = JointMLPTransformer(in_dim, n_mlps, emb_dim, mlp_hidden_layers, mlp_dropout,
                                out_dim, task, tcfg, subspace_cfg=sub_cfg).to(device)
    loss_fn = make_loss(task)
    params = [
        {"params":[p for m in model.mlps for p in m.parameters()], "lr":cfg.lr_mlp},
        {"params": model.agg.parameters(), "lr": cfg.lr_tr},
    ]
    opt = optim.AdamW(params, weight_decay=cfg.weight_decay)

    best, best_state, patience = -1e9, None, cfg.patience
    for ep in range(cfg.epochs):
        if model.subspace_cfg.resample_each_epoch:
            model.resample_feature_subsets()
        model.train()
        for xb,yb in tr:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits,tokens,_ = model(xb)
            if task=="binary":   loss = loss_fn(logits.view(-1), yb.float())
            elif task=="multiclass": loss = loss_fn(logits, yb.long())
            else:                loss = loss_fn(logits.view_as(yb).float(), yb.float())
            if cfg.lambda_div>0: loss = loss + cfg.lambda_div*cka_offdiag_mean(tokens) * 0.0  # keep structure
            if cfg.lambda_cka > 0:
                loss = loss + cfg.lambda_cka * cka_offdiag_mean(tokens)
            loss.backward(); opt.step()

        model.eval(); scores=[]
        with torch.no_grad():
            for xb,yb in va:
                xb,yb = xb.to(device), yb.to(device)
                logits,_,_ = model(xb)
                scores.append(metric(task, logits, yb))
        val = float(np.mean(scores)) if scores else -1e9
        print(f"Epoch {ep+1:02d}  val={val:.6f} {'*' if val>best+1e-6 else ''}")
        if val>best+1e-6:
            best, best_state, patience = val, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, cfg.patience
        else:
            patience -= 1
            if patience<=0: break

    if best_state is not None: model.load_state_dict(best_state)

    model.eval(); scores=[]
    with torch.no_grad():
        for xb,yb in te:
            xb,yb = xb.to(device), yb.to(device)
            logits,_,_ = model(xb)
            scores.append(metric(task, logits, yb))
    test = float(np.mean(scores)) if scores else float("nan")
    print(f"[DONE] best_val={best:.6f} | test={test:.6f}")
    return model, {"best_val": best, "test": test}
