import pyagrum as gum
from enum import Enum, auto

class StructureAlgorithm(Enum):
    HILL_CLIMBING = auto()
    K2 = auto()
    GENETIC_K2 = auto()
    STRUCTURAL_EM = auto()
    PC = auto()
    FCI = auto()
    # RFCI = auto() use PyTetrad

class ParameterAlgorithm(Enum):
    MLE = auto()
    # EM = auto()
    BAYESIAN_DIRICHLET_PRIORS = auto()
    ROBUST_BAYESIAN_ESTIMATE = auto()
    SHRINKAGE_EXIMATOR = auto()

def learn(location: str, structure_algo: StructureAlgorithm,
    parameter_algo: ParameterAlgorithm = ParameterAlgorithm.MLE) -> gum.BayesNet:
    return gum.BayesNet()

def learn_agrum(location: str, parameter_algo: ParameterAlgorithm) -> gum.BayesNet:
    learner: gum.BNLearner = gum.BNLearner(location)
    match parameter_algo:
        case ParameterAlgorithm.BAYESIAN_DIRICHLET_PRIORS:
            learner.useDirichletPrior()
        case ParameterAlgorithm.ROBUST_BAYESIAN_ESTIMATE:
            pass
        case ParameterAlgorithm.SHRINKAGE_EXIMATOR:
            pass
    return learner.learnBN()

def learn_causal(location: str):
    pass

def learn_parameters(location: str, parameter_algo: ParameterAlgorithm):
    pass