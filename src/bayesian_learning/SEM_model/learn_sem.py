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

def learn_structure(location: str, algo: SEMAlgorithm) -> SEMResult:
    """
    Learns the DAG structure from data.
    
    Args:
        location: Path to the dataset (CSV file).
        algo: The SEMAlgorithm to use.
        
    Returns:
        SEMResult (or a specific subclass like GranDAGResult) containing the learned DAG.
    """
    import pandas as pd
    data = pd.read_csv(location)
    
    match algo:
        case SEMAlgorithm.DAGMA:
            wrapper = DagmaWrapper(model_type='nonlinear')
            
        case SEMAlgorithm.DAG_GNN:
            wrapper = DAGGNNWrapper()
            
        case SEMAlgorithm.GRAN_DAG:
            wrapper = GraNDAGWrapper()
            
        case _:
            raise ValueError(f"Unknown algorithm: {algo}")
            
    return wrapper.learn(data)

