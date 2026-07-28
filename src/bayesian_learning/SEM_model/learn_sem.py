from enum import Enum, auto
from .result import SEMResult, GranDAGResult
import numpy as np

from .wrappers.dagma_wrapper import DagmaWrapper
from .wrappers.gran_dag_wrapper import GraNDAGWrapper
from .wrappers.dag_gnn_wrapper import DAGGNNWrapper

class SEMAlgorithm(Enum):
    DAGMA = auto()
    DAG_GNN = auto()
    GRAN_DAG = auto()

def learn_structure(location: str, algo: SEMAlgorithm, **kwargs) -> SEMResult:
    """
    Learns the DAG structure from data.
    
    Args:
        location: Path to the dataset (CSV file).
        algo: The SEMAlgorithm to use.
        **kwargs: Hyperparameters to pass to the underlying models.
        
    Returns:
        SEMResult (or a specific subclass like GranDAGResult) containing the learned DAG.
    """
    import pandas as pd
    data = pd.read_csv(location)
    
    match algo:
        case SEMAlgorithm.DAGMA:
            model_type = kwargs.pop('model_type', 'nonlinear')
            wrapper = DagmaWrapper(model_type=model_type, **kwargs)
            
        case SEMAlgorithm.DAG_GNN:
            wrapper = DAGGNNWrapper(**kwargs)
            
        case SEMAlgorithm.GRAN_DAG:
            wrapper = GraNDAGWrapper(**kwargs)
            
        case _:
            raise ValueError(f"Unknown algorithm: {algo}")
            
    return wrapper.learn(data)

