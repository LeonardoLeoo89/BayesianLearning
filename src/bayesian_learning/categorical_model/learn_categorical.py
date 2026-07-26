from re import match

import pandas as pd
import pyagrum as gum
from enum import Enum, auto
from pytetrad.tools.TetradSearch import TetradSearch
from typing import Any

ALPHA: float = 0.05

class InvalidBranchException(Exception):
    pass

class StructureAlgorithm(Enum):
    HILL_CLIMBING = auto()
    K2 = auto()
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
        case StructureAlgorithm.K2:
            pass
        case StructureAlgorithm.GENETIC_K2:
            pass
        case StructureAlgorithm.STRUCTURAL_EM:
            pass
        case StructureAlgorithm.PC:
            pass
        case StructureAlgorithm.FCI:
            pass
        case StructureAlgorithm.RFCI:
            pass
    return gum.BayesNet()

def learn_agrum(location: str, structure_algo: StructureAlgorithm,
                parameter_algo: ParameterAlgorithm) -> gum.BayesNet:
    learner: gum.BNLearner = gum.BNLearner(location)
    match structure_algo:
        case StructureAlgorithm.HILL_CLIMBING:
            learner.useGreedyHillClimbing()
        case StructureAlgorithm.K2:
            learner.useK2()
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

def learn_tetrad(location: str, structure_algo: StructureAlgorithm) -> Any:
    data: pd.DataFrame = pd.read_csv(location)
    search: TetradSearch = TetradSearch(data)
    search.use_g_square(alpha=ALPHA)
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
    return learner.learnParameters(structure)