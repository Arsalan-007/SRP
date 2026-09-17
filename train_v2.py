import torch
import torch.nn as nn
from weight_schemes import WeightScheme


def _compute_step_weights(num_steps, device, scheme='exponential', **kwargs):
    """
    Compute weights for each step using the specified scheme.
    
    Args:
        num_steps: Number of predictors (K)
        device: Device to place weights on
        scheme: One of 'linear', 'uniform', 'exponential'
        **kwargs: Additional arguments passed to WeightScheme
        
    Returns:
        Normalized weights tensor of shape (num_steps,)
    """
    weight_scheme = WeightScheme(method=scheme, **kwargs)
    return weight_scheme.get_weights(num_steps, device)


def _step_loss(criterion, step_pred, y_batch, task):
    if task == 'regression':
        return criterion(step_pred.view_as(y_batch).float(), y_batch.float())
    return criterion(step_pred, y_batch.long())


def train(model, X_tr, y_tr, X_val, y_val, epochs=50, lr=1e-3, batch_size=64, 
          l1_lambda=1e-4, l2_lambda=1e-4, weight_scheme='exponential', **weight_kwargs):
    """
    Training loop with weighted losses for each predictor.
    
    Args:
        model: SRP model to train
        X_tr, y_tr: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        l1_lambda: L1 regularization strength
        l2_lambda: L2 regularization strength (weight decay)
        weight_scheme: Weighting scheme - 'linear', 'uniform', or 'exponential'
        **weight_kwargs: Additional arguments for WeightScheme (e.g., base for exponential)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
    
    if model.task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    history = {'train_loss': [], 'val_loss': []}
    num_batches = int(torch.ceil(torch.tensor(len(X_tr) / batch_size)))
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        indices = torch.randperm(len(X_tr))
        X_tr_shuffled = X_tr[indices]
        y_tr_shuffled = y_tr[indices]
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_tr_shuffled[start_idx:end_idx]
            y_batch = y_tr_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            predictions, features, _ = model(X_batch)
            K = predictions.shape[1]
            weights = _compute_step_weights(K, predictions.device, weight_scheme, **weight_kwargs)
            task_loss = predictions.new_tensor(0.0)
            
            # Weighted sum of losses
            for i in range(K):
                step_pred = predictions[:, i]
                step_loss = _step_loss(criterion, step_pred, y_batch, model.task)
                task_loss += weights[i] * step_loss
            
            # L1 penalty on weak learner weights to encourage sparsity.
            l1_penalty = model.ensemble.l1_penalty(normalize=True, include_bias=False)
            loss = task_loss + (l1_lambda * l1_penalty)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            val_preds, _, _ = model(X_val)
            K = val_preds.shape[1]
            weights = _compute_step_weights(K, val_preds.device, weight_scheme, **weight_kwargs)
            val_loss = val_preds.new_tensor(0.0)
            
            for i in range(K):
                step_pred = val_preds[:, i]
                step_loss = _step_loss(criterion, step_pred, y_val, model.task)
                val_loss += weights[i] * step_loss
                
            history['val_loss'].append(val_loss.item())
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss.item():.4f}")
            
    return history