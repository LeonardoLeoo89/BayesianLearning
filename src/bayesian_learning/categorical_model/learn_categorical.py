import pyagrum as gum
from enum import Enum, auto

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
    EM = auto()
    BAYESIAN_DIRICHLET_PRIORS = auto()
    ROBUST_BAYESIAN_ESTIMATE = auto()
    SHRINKAGE_EXIMATOR = auto()
    UNSPECIFIED = auto()

def learn(location: str, structure_algo: StructureAlgorithm,
    parameter_algo: ParameterAlgorithm = ParameterAlgorithm.UNSPECIFIED) -> gum.BayesNet:
    return gum.BayesNet()

def learn_structure(location: str, algo: StructureAlgorithm) -> gum.BayesNet:
    learner: gum.BNLearner = gum.BNLearner(location)

    match algo:
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

def learn_parameters(data, net: gum.BayesNet, algo: ParameterAlgorithm):
    pass
