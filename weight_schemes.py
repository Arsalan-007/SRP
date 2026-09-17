"""
Weight schemes module for SRP project.
Implements various weighting strategies for multi-step predictor training.
"""

import torch
import numpy as np
from typing import Optional


class WeightScheme:
    """
    Unified weight scheme selector for multi-step predictors.
    
    Supports:
    - 'linear': Higher weight for later predictors (linearly increasing)
    - 'uniform': Equal weight for all predictors
    - 'exponential': Higher weight for later predictors (exponentially increasing)
    """
    
    def __init__(self, method: str = 'exponential', **kwargs):
        """
        Initialize weight scheme with specified method.
        
        Args:
            method: One of 'linear', 'uniform', 'exponential'
            **kwargs: Additional arguments (e.g., base for exponential)
        """
        self.method = method
        self.kwargs = kwargs
        
    def get_weights(self, num_steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute weights for each step.
        
        Args:
            num_steps: Number of predictors (K)
            device: Device to place weights on
            
        Returns:
            Normalized weights tensor of shape (num_steps,)
        """
        if self.method == 'linear':
            # Linearly increasing weights: first predictor gets lowest weight, last gets highest
            # This is appropriate for causal attention where later predictors see more context
            weights = torch.arange(1, num_steps + 1, dtype=torch.float32, device=device)
            #fix this to start from 0: weights = torch.arange(num_steps, dtype=torch.float32, device=device) + 1
            weights = torch.arange(num_steps, dtype=torch.float32, device=device) + 1
        elif self.method == 'uniform':
            # Equal weights for all steps
            weights = torch.ones(num_steps, dtype=torch.float32, device=device)
        elif self.method == 'exponential':
            # Higher weight for later steps: [2^0, 2^1, ..., 2^(K-1)]
            base = self.kwargs.get('base', 2.0)
            indices = torch.arange(num_steps, dtype=torch.float32, device=device)
            weights = torch.pow(base, indices)
        else:
            raise ValueError(f"Unknown weight scheme: {self.method}")
        
        return weights / weights.sum()
    
    def __repr__(self):
        return f"WeightScheme(method='{self.method}', kwargs={self.kwargs})"