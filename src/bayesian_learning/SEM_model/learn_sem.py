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

def learn(location: str, algo: SEMAlgorithm) -> SEMResult:
    """
    Convenience function to learn both structure and parameters (if applicable).
    For SEMs, learning structure and parameters often happens simultaneously 
    during neural network training.
    """
    return learn_structure(location, algo)

def learn_structure(location: str, algo: SEMAlgorithm) -> SEMResult:
    """
    Learns the DAG structure from data.
    
    Args:
        location: Path to the dataset (e.g., CSV file).
        algo: The SEMAlgorithm to use.
        
    Returns:
        SEMResult (or a specific subclass like GranDAGResult) containing the learned DAG.
    """
    import pandas as pd
    data = pd.read_csv(location)
    
    match algo:
        case SEMAlgorithm.DAGMA:
            wrapper = DagmaWrapper(model_type='linear') # Defaults to linear for now
            
        case SEMAlgorithm.DAG_GNN:
            wrapper = DAGGNNWrapper()
            
        case SEMAlgorithm.GRAN_DAG:
            wrapper = GraNDAGWrapper()
            
        case _:
            raise ValueError(f"Unknown algorithm: {algo}")
            
    return wrapper.learn(data)

def learn_parameters(data, net: SEMResult, algo: SEMAlgorithm):
    """
    Fits parameters to a given DAG structure. 
    Note: Some algorithms (like GraN-DAG) fit these during structure learning 
    and you can query them directly from the result object (e.g. net.predict_distribution()).
    """
    match algo:
        case SEMAlgorithm.GRAN_DAG:
            print("GraN-DAG parameters are fitted during structure learning.")
            if isinstance(net, GranDAGResult):
                print("Use net.predict_distribution() to query distributions.")
            else:
                print("Error: Expected GranDAGResult for GRAN_DAG parameters.")
        case _:
            pass
