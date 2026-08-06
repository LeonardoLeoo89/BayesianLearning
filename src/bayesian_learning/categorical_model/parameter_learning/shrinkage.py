import pyagrum as gum
import pandas as pd
from pandas.core.series import Series
import numpy as np
from typing import Any

def learn_shrinkage_parameters(location: str, bn: Any) -> gum.BayesNet:
    """Shrinkage estimator for parameter learning.

        Applies the shrinkage estimator on an already
        defined structure and a dataset to learn the
        bayesian network's CPTS

        Args:
            location: The path of the dataset (csv).
            bn: A bayesian network with an already defined structure.

        Returns:
            The bayesian network with updated CPTS.
        """
    df: pd.DataFrame = pd.read_csv(location)
    for col in df.columns:
        df[col] = df[col].astype(str)

    node_name: str
    parents: list[str]
    parent_vals: tuple
    parent_dict: dict[str, str]
    r_i: int
    t_ijk: float
    cpt: gum.Tensor
    labels: list[str]
    counts: Series[int]
    n_i: int
    n_ij: int
    ml_estimates: np.ndarray
    num: float
    den: float
    lambda_star: float
    shrink_probs: np.ndarray

    for node_id in bn.nodes():
        node_name = bn.variable(node_id).name()
        parents = [bn.variable(p).name() for p in bn.parents(node_id)]
        r_i = bn.variable(node_id).domainSize()
        t_ijk = 1.0 / r_i
        cpt = bn.cpt(node_id)
        cpt.fillWith(t_ijk)
        labels = bn.variable(node_id).labels()
        
        if not parents:
            counts = df[node_name].value_counts()
            n_i = counts.sum()
            
            if n_i > 1:
                ml_estimates = np.array([counts.get(label, 0) / n_i for label in labels])
                
                num = 1.0 - np.sum(ml_estimates**2)
                den = (n_i - 1) * np.sum((t_ijk - ml_estimates)**2)
                lambda_star = num / den if den > 0 else 1.0
                lambda_star = min(1.0, max(0.0, lambda_star))
                
                shrink_probs = lambda_star * t_ijk + (1 - lambda_star) * ml_estimates
                cpt[{}] = list(shrink_probs)
                
        else:
            group_counts = df.groupby(parents)[node_name].value_counts().unstack(fill_value=0)
            
            for index_val, row in group_counts.iterrows():
                if isinstance(index_val, tuple):
                    parent_vals = index_val
                else:
                    parent_vals = (index_val,)
                    
                parent_dict = {parent: str(val) for parent, val in zip(parents, parent_vals)}
                
                n_ij = row.sum()
                if n_ij > 1:
                    ml_estimates = np.array([row.get(label, 0) / n_ij for label in labels])
                    
                    num = 1.0 - np.sum(ml_estimates**2)
                    den = (n_ij - 1) * np.sum((t_ijk - ml_estimates)**2)
                    lambda_star = num / den if den > 0 else 1.0
                    lambda_star = min(1.0, max(0.0, lambda_star))
                    
                    shrink_probs = lambda_star * t_ijk + (1 - lambda_star) * ml_estimates

                    cpt[parent_dict] = list(shrink_probs)
    return bn
