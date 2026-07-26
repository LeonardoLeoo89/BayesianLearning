import pyagrum as gum
import pandas as pd
from typing import List, Dict, Set, Any

def generate_ess_dataset(bn: gum.BayesNet, df: pd.DataFrame) -> pd.DataFrame:
    """
    E-step: Creates a weighted, completed dataset from partially observed data.
    """
    ie: gum.LazyPropagation = gum.LazyPropagation(bn)
    completed_rows: List[Dict[str, Any]] = []
    variables: tuple = bn.names()
    
    # Iterate through every row in our partially observed dataset
    for idx, row in df.iterrows():
        observed: Dict[str, int] = {}
        missing: Set[str] = set()
        
        # Sort variables into observed vs missing for this row
        for var in variables:
            val = row[var]
            if pd.isna(val) or val == '?' or val == '':
                missing.add(var)
            else:
                observed[var] = int(val) # Cast to int assuming discrete categories
                
        # If the row is fully observed, we just keep it with a weight of 1
        if not missing:
            row_dict = observed.copy()
            row_dict['_weight'] = 1.0
            completed_rows.append(row_dict)
            continue
            
        # If there are missing variables, compute their joint posterior probability
        ie.setEvidence(observed)
        ie.makeInference()
        jp: gum.Potential = ie.jointPosterior(missing)
        
        # Iterate over all possible configurations of the missing variables
        inst: gum.Instantiation = gum.Instantiation(jp)
        inst.setFirst()
        while not inst.end():
            prob = jp.get(inst)
            
            # We only care about configurations with non-zero probability
            if prob > 0:
                row_dict = observed.copy()
                # Fill in the missing values with this specific configuration
                for var in jp.names:
                    row_dict[var] = inst.val(var)
                # The expected sufficient statistic is the probability (weight)
                row_dict['_weight'] = prob
                completed_rows.append(row_dict)
                
            inst.inc()
            
    # Return as a new completed DataFrame
    return pd.DataFrame(completed_rows)


def structural_em(df_missing: pd.DataFrame, initial_bn: gum.BayesNet, max_iters: int = 10, epsilon: float = 1e-3, score: str = 'bic') -> gum.BayesNet:
    """
    Structural EM Algorithm loop.
    """
    current_bn: gum.BayesNet = initial_bn
    
    for t in range(max_iters):
        
        # ---------------------------------------------------------
        # Step 3: Optional Parameter Learning (EM for parameters)
        # ---------------------------------------------------------
        learner_param: gum.BNLearner = gum.BNLearner(df_missing)
        learner_param.useEM(epsilon)
        current_bn = learner_param.learnParameters(current_bn.dag())
        
        # ---------------------------------------------------------
        # Step 4: Generate Expected Sufficient Statistics (E-Step)
        # ---------------------------------------------------------
        df_ess: pd.DataFrame = generate_ess_dataset(current_bn, df_missing)
        
        # ---------------------------------------------------------
        # Step 5 & 6: Structure Learn & Estimate Parameters (M-Step)
        # ---------------------------------------------------------
        # Create a clean dataframe for the learner without the weight column
        df_clean: pd.DataFrame = df_ess.drop(columns=['_weight'])
        learner_struct: gum.BNLearner = gum.BNLearner(df_clean)
        
        # Apply the expected sufficient statistics as row weights
        for i, w in enumerate(df_ess['_weight']):
            learner_struct.setRecordWeight(i, float(w))
            
        # Configure the structure learner
        learner_struct.useLocalSearchWithTabuList()
        if score.lower() == 'bic':
            learner_struct.useScoreBIC()
        elif score.lower() == 'bdeu':
            learner_struct.useScoreBDeu()
            
        # Learn the new DAG and parameters
        new_dag: gum.DAG = learner_struct.learnDAG()
        new_bn: gum.BayesNet = learner_struct.learnParameters(new_dag)
        
        # Check for structural convergence (if the DAG hasn't changed)
        if new_dag.arcs() == current_bn.dag().arcs():
            current_bn = new_bn
            break
            
        current_bn = new_bn
        
    return current_bn
