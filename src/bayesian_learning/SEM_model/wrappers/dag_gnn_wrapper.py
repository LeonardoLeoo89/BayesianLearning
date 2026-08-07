import pandas as pd
from ..result import SEMResult
from .base import SEMWrapper
from dag_gnn.trainer import DAGGNNTrainer # type: ignore

class DAGGNNWrapper(SEMWrapper):
    """Wrapper for the DAG-GNN algorithm."""

    def __init__(self, **kwargs):
        """Initializes the DAG-GNN wrapper.
        
        Args:
            **kwargs: Hyperparameters to pass to DAG-GNN (e.g. epochs=300, lr=3e-3).
        """
        self.kwargs = kwargs

    def learn(self, data: pd.DataFrame) -> SEMResult:
        """Learns the DAG structure using DAG-GNN."""
        import torch
        
        # Ensure CUDA is enabled if available, unless explicitly disabled by user
        if 'cuda' not in self.kwargs:
            self.kwargs['cuda'] = torch.cuda.is_available()
            
        print(f"DAG-GNN Wrapper: CUDA enabled = {self.kwargs['cuda']}")

        trainer = DAGGNNTrainer(**self.kwargs)
        W_est = trainer.fit(data.values)

        return SEMResult(W_est, node_names=list(data.columns))
