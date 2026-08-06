from enum import Enum, auto
from typing import Any

class InvalidBranchException(Exception):
    pass

class StructureAlgorithm(Enum):
    HILL_CLIMBING = auto()
    GENETIC_K2 = auto()
    STRUCTURAL_EM = auto()
    PC = auto()
    FCI = auto()
    RFCI = auto()

class ParameterAlgorithm(Enum):
    MLE = auto()
    EM = auto()
    BAYESIAN_DIRICHLET_PRIORS = auto()
    SHRINKAGE_ESTIMATOR = auto()

def learn(location: str, structure_algo: StructureAlgorithm,
    parameter_algo: ParameterAlgorithm = ParameterAlgorithm.MLE) -> Any:
    import pyagrum as gum
    if structure_algo == StructureAlgorithm.STRUCTURAL_EM and parameter_algo != ParameterAlgorithm.EM:
        raise InvalidBranchException("STRUCTURAL_EM must be used with ParameterAlgorithm.EM")

    dag: gum.DAG
    match structure_algo:
        case StructureAlgorithm.HILL_CLIMBING:
            dag = learn_agrum_structure(location, structure_algo)
        case StructureAlgorithm.GENETIC_K2:
            from bayesian_learning.categorical_model.genetic_K2.genetic_K2 import genetic_k2
            dag = genetic_k2(location)
        case StructureAlgorithm.STRUCTURAL_EM:
            from bayesian_learning.categorical_model.structural_em.structural_em import structural_em
            import pandas as pd
            initial_bn = gum.BayesNet()
            df: pd.DataFrame = pd.read_csv(location)
            for col in df.columns:
                initial_bn.add(gum.LabelizedVariable(col, col, [str(v) for v in df[col].dropna().unique()]))
            return structural_em(location, initial_bn)
        case StructureAlgorithm.PC | StructureAlgorithm.FCI | StructureAlgorithm.RFCI:
            from bayesian_learning.categorical_model.categorical_translator.tetrad2agrum import translate
            tetrad_dag = learn_tetrad(location, structure_algo)
            dag = translate(tetrad_dag).dag()
    return learn_parameters(location, dag, parameter_algo)

def learn_agrum_structure(location: str, structure_algo: StructureAlgorithm) -> Any:
    import pyagrum as gum
    learner: gum.BNLearner = gum.BNLearner(location)
    match structure_algo:
        case StructureAlgorithm.HILL_CLIMBING:
            learner.useGreedyHillClimbing()
        case _:
            raise InvalidBranchException(f"Unexpected algorithm \"{structure_algo}\" for pyAgrum learning")
    return learner.learnDAG()

def learn_tetrad(location: str, structure_algo: StructureAlgorithm,
                 alpha: float = 0.05) -> Any:
    import pandas as pd
    from pytetrad.tools.TetradSearch import TetradSearch
    data: pd.DataFrame = pd.read_csv(location)
    search: TetradSearch = TetradSearch(data)
    search.use_g_square(alpha=alpha)
    match structure_algo:
        case StructureAlgorithm.PC:
            search.run_pc()
        case StructureAlgorithm.FCI:
            search.run_fci()
        case StructureAlgorithm.RFCI:
            search.run_rfci()
    return search.get_java()

def learn_parameters(location: str, dag: Any,
                     parameter_algo: ParameterAlgorithm) -> Any:
    import pyagrum as gum
    learner: gum.BNLearner = gum.BNLearner(location)

    match parameter_algo:
        case ParameterAlgorithm.EM:
            learner.useEM(1e-3)
            learner.useSmoothingPrior(1.0)
        case ParameterAlgorithm.BAYESIAN_DIRICHLET_PRIORS:
            learner.useBDeuPrior(1.0)
        case ParameterAlgorithm.MLE:
            learner.useSmoothingPrior(1e-4)
        case ParameterAlgorithm.SHRINKAGE_ESTIMATOR:
            learner.useSmoothingPrior(1e-4)
            bn = learner.learnParameters(dag)
            from bayesian_learning.categorical_model.parameter_learning.shrinkage import learn_shrinkage_parameters
            return learn_shrinkage_parameters(location, bn)
            
    return learner.learnParameters(dag)