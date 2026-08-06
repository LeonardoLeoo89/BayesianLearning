from typing import List, Dict, Set, Any
import pyagrum as gum
import pandas as pd

def generate_ess_dataset(bn: gum.BayesNet, df: pd.DataFrame) -> pd.DataFrame:
    """ESS dataset generator.

        Generates a dataset weighted by the
        expected sufficient statistics calculated from a
        given dataset and an already defined bayesian network.

        Args:
            bn: A Bayesian network.
            df: A dataframe.

        Returns:
            The ESS-based dataframe.
        """
    ie: gum.LazyPropagation = gum.LazyPropagation(bn)
    completed_rows: List[Dict[str, Any]] = []
    variables: tuple[str, ...] = bn.names() # type: ignore
    observed: Dict[str, Any]
    missing: Set[str]
    jp: gum.Tensor
    inst: gum.Instantiation
    prob: float

    for idx, row in df.iterrows():
        observed = {}
        missing = set()
        
        for var in variables:
            val = row[var]
            if pd.isna(val) or val == '?' or val == '':
                missing.add(var)
            else:
                observed[var] = str(val)

        if not missing:
            row_dict = observed.copy()
            row_dict["_weight"] = 1.0
            completed_rows.append(row_dict)
            continue
            
        ie.setEvidence(observed)
        ie.addJointTarget(missing)
        ie.makeInference()
        jp = ie.jointPosterior(missing)
        
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
                  max_iters: int = 100,
                  epsilon: float | int = 1e-3) -> gum.BayesNet:
    """Structural EM algorithm.

        Implementation of the EM concept for bayesian network learning,
        capable of handling missing data.
        The algorithm alternates between an Expectation phase (ESS computation)
        and a Maximization phase (network learning)

        Args:
            location: The path of the dataset (csv).
            initial_bn: Starting bayesian network.
            max_iters: The number of maximum iterations.
                Defaults to 100.
            epsilon: The epsilon parameter for parameter EM.
                Defaults to 1e-3.

        Returns:
            The learned bayesian network.
        """
    current: gum.BayesNet = initial_bn
    df_missing: pd.DataFrame = pd.read_csv(location)
    t: int = 0
    learner: gum.BNLearner
    df_ess: pd.DataFrame
    df_clean: pd.DataFrame
    learner_struct: gum.BNLearner
    new_dag: gum.DAG
    new_bn: gum.BayesNet

    for t in range(max_iters):
        
        # parameter EM
        learner = gum.BNLearner(df_missing)
        learner.useEM(epsilon)
        learner.useSmoothingPrior(1.0)
        current = learner.learnParameters(current.dag())

        # e-step
        df_ess = generate_ess_dataset(current, df_missing)
        
        # m-step
        df_clean = df_ess.drop(columns=['_weight'])
        learner_struct = gum.BNLearner(df_clean)
        for i, w in enumerate(df_ess['_weight']):
            learner_struct.setRecordWeight(i, float(w))

        learner_struct.useLocalSearchWithTabuList()
        learner_struct.useSmoothingPrior(1.0)
        new_dag = learner_struct.learnDAG()
        new_bn = learner_struct.learnParameters(new_dag)

        if new_dag.arcs() == current.dag().arcs():
            break
        current = new_bn
        
    return current
