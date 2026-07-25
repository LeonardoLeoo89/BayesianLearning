import pandas as pd
import numpy as np
import torch
from ..result import SEMResult
from .base import SEMWrapper

class DagmaWrapper(SEMWrapper):
    """Wrapper for the pip-installed dagma library."""
    
    def __init__(self, model_type: str = 'linear', **kwargs):
        """
        Args:
            model_type: 'linear' or 'nonlinear' (MLP)
            **kwargs: Additional parameters for the Dagma model (e.g. lambda1)
        """
        self.model_type = model_type
        self.kwargs = kwargs
        
    def learn(self, data: pd.DataFrame) -> SEMResult:
        """Learns the DAG structure using DAGMA."""
        
        # We need to import locally so that if a user doesn't have it installed,
        # it doesn't crash the whole module on import.
        if self.model_type == 'linear':
            from dagma.linear import DagmaLinear
            model = DagmaLinear(loss_type='l2')
        elif self.model_type == 'nonlinear':
            from dagma.nonlinear import DagmaMLP, DagmaNonlinear
            d = data.shape[1]
            eq_model = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=torch.double)
            model = DagmaNonlinear(eq_model, dtype=torch.double)
        else:
            raise ValueError(f"Unknown Dagma model_type: {self.model_type}")
            
        # Convert DataFrame to numpy for training
        X = data.values
        
        # Fit model
        W_est = model.fit(X, **self.kwargs)
        
        # Return universal result
        return SEMResult(W_est, node_names=list(data.columns))
