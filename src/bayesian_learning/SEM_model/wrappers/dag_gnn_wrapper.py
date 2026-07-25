import pandas as pd
from ..result import SEMResult
from .base import SEMWrapper
from dag_gnn.trainer import DAGGNNTrainer # type: ignore

class DAGGNNWrapper(SEMWrapper):
    """Wrapper for the DAG-GNN algorithm."""
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Hyperparameters to pass to DAG-GNN (e.g. epochs=300, lr=3e-3)
        """
        self.kwargs = kwargs
        
    def learn(self, data: pd.DataFrame) -> SEMResult:
        """Learns the DAG structure using DAG-GNN."""
        
        # Instantiate the trainer with user-provided hyperparameters
        trainer = DAGGNNTrainer(**self.kwargs)
        
        # Run the PyTorch loop directly in-memory!
        W_est = trainer.fit(data.values)
        
        return SEMResult(W_est, node_names=list(data.columns))
