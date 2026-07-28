import pyagrum as gum
import pandas as pd
from typing import List, Dict, Set, Any

def generate_ess_dataset(bn: gum.BayesNet, df: pd.DataFrame) -> pd.DataFrame:
    ie: gum.LazyPropagation = gum.LazyPropagation(bn)
    completed_rows: List[Dict[str, Any]] = []
    variables: tuple[str, ...] = bn.names() # type: ignore

    observed: Dict[str, float | int]
    missing: Set[str]
    prob: float
    row_dict: Dict[str, float | int]
    inst: gum.Instantiation
    jp: gum.Tensor
    for idx, row in df.iterrows():
        observed= {}
        missing = set()
        
        for var in variables:
            val = row[var]
            if pd.isna(val) or val == '?' or val == '':
                missing.add(var)
            else:
                observed[var] = int(val) # type: ignore

        if not missing:
            row_dict = observed.copy()
            row_dict["_weight"] = 1.0
            completed_rows.append(row_dict)
            continue
            
        # If there are missing variables, compute their joint posterior probability
        ie.setEvidence(observed)
        ie.makeInference()
        jp = ie.jointPosterior(missing)
        
        # Iterate over all possible configurations of the missing variables
        inst = gum.Instantiation(jp)
        inst.setFirst()
        while not inst.end():
            prob = jp.get(inst)

            if prob > 0:
                row_dict = observed.copy()
                for var in jp.names:
                    row_dict[var] = inst.val(var)
                row_dict["_weight"] = prob
                completed_rows.append(row_dict)
                
            inst.inc()


    return pd.DataFrame(completed_rows)


def structural_em(location: str, initial_bn: gum.BayesNet,
                  max_iters: float | int = float("inf"),
                  epsilon: float | int = 1e-3) -> gum.BayesNet:
    """
    Structural EM Algorithm loop.
    """
    current: gum.BayesNet = initial_bn
    learner: gum.BNLearner
    new_dag: gum.DAG
    new_bn: gum.BayesNet
    df_missing: pd.DataFrame = pd.read_csv(location)
    t: int = 0
    while t < max_iters:
        
        # parameter EM
        learner = gum.BNLearner(df_missing)
        learner.useEM(epsilon)
        current = learner.learnParameters(current.dag())

        # e-step
        df_ess: pd.DataFrame = generate_ess_dataset(current, df_missing)
        
        # m-step
        df_clean: pd.DataFrame = df_ess.drop(columns=['_weight'])
        learner_struct: gum.BNLearner = gum.BNLearner(df_clean)
        for i, w in enumerate(df_ess['_weight']):
            learner_struct.setRecordWeight(i, float(w))

        learner_struct.useLocalSearchWithTabuList()
        new_dag = learner_struct.learnDAG()
        new_bn = learner_struct.learnParameters(new_dag)

        if new_dag.arcs() == current.dag().arcs():
            break
        current = new_bn
        t += 1
        
    return current
