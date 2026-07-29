import pyagrum as gum
import pandas as pd
import numpy as np
from typing import Any

def learn_shrinkage_parameters(location: str, bn: Any) -> Any:
    """
    Learns the Conditional Probability Tables (CPTs) of a Bayesian Network using 
    the James-Stein Shrinkage Estimator (Hausser and Strimmer, 2009).
    This adaptively shrinks the high-variance Maximum Likelihood (ML) estimates 
    towards a low-variance uniform target based on the local sample size of 
    each parent configuration.
    """
    df = pd.read_csv(location)
    
    # Ensure categorical columns are treated as strings
    for col in df.columns:
        df[col] = df[col].astype(str)

    for node_id in bn.nodes():
        node_name = bn.variable(node_id).name()
        parents = [bn.variable(p).name() for p in bn.parents(node_id)]
        
        r_i = bn.variable(node_id).domainSize()
        t_ijk = 1.0 / r_i
        
        cpt = bn.cpt(node_id)
        # Initialize the entire CPT with the uniform target probability
        cpt.fillWith(t_ijk)
        
        labels = bn.variable(node_id).labels()
        
        if not parents:
            # Root node (marginal probabilities)
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
            # Child node (conditional probabilities)
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
                    
                    try:
                        cpt[parent_dict] = list(shrink_probs)
                    except gum.pyagrum.InvalidArgument:
                        # This happens if a parent configuration in the dataset has a value 
                        # that was never registered in the domain of the BayesNet variable
                        pass

    return bn
