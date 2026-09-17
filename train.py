import torch
import torch.nn as nn


def _deep_supervision_weights(num_steps, device):
    weights = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
    return weights / weights.sum()


def _step_loss(criterion, step_pred, y_batch, task):
    if task == 'regression':
        return criterion(step_pred.view_as(y_batch).float(), y_batch.float())
    return criterion(step_pred, y_batch.long())


def train(model, X_tr, y_tr, X_val, y_val, epochs=50, lr=1e-3, batch_size=64, l1_lambda=1e-4, l2_lambda=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
    
    # Standard scalar loss (MSE for regression, CrossEntropy for classification)
    if model.task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    history = {'train_loss': [], 'val_loss': []}
    num_batches = int(torch.ceil(torch.tensor(len(X_tr) / batch_size)))
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_tr))
        X_tr_shuffled = X_tr[indices]
        y_tr_shuffled = y_tr[indices]
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_tr_shuffled[start_idx:end_idx]
            y_batch = y_tr_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # predictions shape: (Batch, K)
            predictions, features, _ = model(X_batch)
            K = predictions.shape[1]
            weights = _deep_supervision_weights(K, predictions.device)
            task_loss = predictions.new_tensor(0.0)
            
            # --- DEEP SUPERVISION LOSS WEIGHTING ---
            for i in range(K):
                step_pred = predictions[:, i]
                step_loss = _step_loss(criterion, step_pred, y_batch, model.task)
                
                # Later steps see more previous learners, so train/evaluate them more.
                task_loss += weights[i] * step_loss
            
            # L2 is handled by AdamW weight_decay; keep L1 explicit on weak learner weights.
            l1_penalty = model.ensemble.l1_penalty(normalize=True, include_bias=False)
            loss = task_loss + (l1_lambda * l1_penalty)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation Loop
        model.eval()
        with torch.no_grad():
            val_preds, _, _ = model(X_val)
            K = val_preds.shape[1]
            weights = _deep_supervision_weights(K, val_preds.device)
            val_loss = val_preds.new_tensor(0.0)
            
            # Compute validation loss with the same weighting as training.
            for i in range(K):
                step_pred = val_preds[:, i]
                step_loss = _step_loss(criterion, step_pred, y_val, model.task)
                val_loss += weights[i] * step_loss
                
            history['val_loss'].append(val_loss.item())
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss.item():.4f}")
            
    return history
