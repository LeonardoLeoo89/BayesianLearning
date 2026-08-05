import numpy as np
import pandas as pd
from ..result import SEMResult, GranDAGResult
from .base import SEMWrapper

class GraNDAGWrapper(SEMWrapper):
    """Wrapper for the GraN-DAG model running natively within the process."""
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Hyperparameters to pass to GraN-DAG (e.g. num_train_iter=100000, lr=1e-3, etc.)
                      Defaults are set internally if not provided.
        """
        self.kwargs = kwargs
        
    def learn(self, data: pd.DataFrame) -> SEMResult:
        """Learns the DAG structure using GraN-DAG."""
        
        # We import here so we don't fail immediately if GraN-DAG is not installed
        from gran_dag.trainer import GraNDAGTrainer
        import torch
        
        n_samples, n_vars = data.shape
        # Ensure GPU is enabled if available, unless explicitly disabled by user
        if 'gpu' not in self.kwargs:
            self.kwargs['gpu'] = torch.cuda.is_available()
            
        print(f"GraN-DAG Wrapper: GPU enabled = {self.kwargs['gpu']}")

        # Instantiate the native OOP trainer
        trainer = GraNDAGTrainer(**self.kwargs)
        
        # Run natively without subprocess or tempfile overhead (handled internally)
        model = trainer.fit(data.values, adjacency_array=np.zeros((n_vars, n_vars)))
        
        # The trainer modifies model.adjacency in-place during the to-dag step
        W_est = model.adjacency.detach().cpu().numpy()
        
        if W_est.shape != (n_vars, n_vars):
            W_est = W_est.reshape((n_vars, n_vars))
            
        return GranDAGResult(W_est, model, node_names=list(data.columns))
