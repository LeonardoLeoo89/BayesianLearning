from enum import Enum, auto
from .result import SEMResult, GranDAGResult
from .wrappers import DagmaWrapper, GraNDAGWrapper, DAGGNNWrapper, SEMWrapper

class SEMAlgorithm(Enum):
    DAGMA = auto()
    DAG_GNN = auto()
    GRAN_DAG = auto()

def learn_structure(location: str, algo: SEMAlgorithm, **kwargs) -> SEMResult:
    """Learns the DAG structure from data.
    
    Args:
        location: Path to the dataset (CSV file).
        algo: The SEMAlgorithm to use.
        **kwargs: Hyperparameters to pass to the underlying models.
        
    Returns:
        SEMResult (or a specific subclass like GranDAGResult) containing the learned DAG.
    """
    import pandas as pd
    data = pd.read_csv(location)
    
    wrapper: SEMWrapper
    
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

