import pandas as pd
import pyagrum as gum
from enum import Enum, auto
from typing import Any
from pytetrad.tools.TetradSearch import TetradSearch
from bayesian_learning.categorical_model.genetic_K2.genetic_K2 import genetic_k2

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
    # EM = auto()
    BAYESIAN_DIRICHLET_PRIORS = auto()
    ROBUST_BAYESIAN_ESTIMATE = auto()
    SHRINKAGE_EXIMATOR = auto()

def learn(location: str, structure_algo: StructureAlgorithm,
    parameter_algo: ParameterAlgorithm = ParameterAlgorithm.MLE) -> gum.BayesNet:
    match structure_algo:
        case StructureAlgorithm.HILL_CLIMBING:
            pass
        case StructureAlgorithm.GENETIC_K2:
            bn, _ = genetic_k2(location)
            return learn_parameters(location, bn, parameter_algo)
        case StructureAlgorithm.STRUCTURAL_EM:
            from bayesian_learning.categorical_model.structural_em.structural_em import structural_em
            # We need an initial BayesNet. An empty one is usually a good start.
            import pyagrum as gum
            initial_bn = gum.BayesNet()
            # We need variables to be initialized in the empty BN to match the dataset
            import pandas as pd
            df = pd.read_csv(location)
            for col in df.columns:
                initial_bn.add(gum.LabelizedVariable(col, col, [str(v) for v in df[col].dropna().unique()]))
            return structural_em(location, initial_bn)
        case StructureAlgorithm.PC | StructureAlgorithm.FCI | StructureAlgorithm.RFCI:
            learn_tetrad(location, structure_algo, ) #TODO
    return gum.BayesNet()

def learn_agrum(location: str, structure_algo: StructureAlgorithm,
                parameter_algo: ParameterAlgorithm) -> gum.BayesNet:
    learner: gum.BNLearner = gum.BNLearner(location)
    match structure_algo:
        case StructureAlgorithm.HILL_CLIMBING:
            learner.useGreedyHillClimbing()
        case _:
            raise InvalidBranchException(f"Unexpected algorithm \"{structure_algo}\"for pyAgrum learning")
    match parameter_algo:
        case ParameterAlgorithm.BAYESIAN_DIRICHLET_PRIORS:
            learner.useDirichletPrior()
        case ParameterAlgorithm.ROBUST_BAYESIAN_ESTIMATE:
            pass
        case ParameterAlgorithm.SHRINKAGE_EXIMATOR:
            pass
    return learner.learnBN()

def learn_tetrad(location: str, structure_algo: StructureAlgorithm,
                 alpha: float = 0.05) -> Any:
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
    return search.get_dag_java()

def learn_parameters(location: str, structure: gum.BayesNet,
                     parameter_algo: ParameterAlgorithm) -> gum.BayesNet:
    learner: gum.BNLearner = gum.BNLearner(location)
    match parameter_algo:
        case ParameterAlgorithm.BAYESIAN_DIRICHLET_PRIORS:
            learner.useDirichletPrior()
        case ParameterAlgorithm.ROBUST_BAYESIAN_ESTIMATE:
            pass
        case ParameterAlgorithm.SHRINKAGE_EXIMATOR:
            pass
    return learner.learnParameters(structure.dag())