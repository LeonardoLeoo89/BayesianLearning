from abc import ABC, abstractmethod
import pandas as pd
from ..result import SEMResult

class SEMWrapper(ABC):
    """Abstract base class for all SEM algorithm wrappers."""
    
    @abstractmethod
    def learn(self, data: pd.DataFrame) -> SEMResult:
        """Learns the DAG structure from data.
        
        Args:
            data: A pandas DataFrame containing the observational data.
            
        Returns:
            A SEMResult (or subclass) containing the learned DAG.
        """
        pass
