import traceback
from bayesian_learning.categorical_model.learn_categorical import learn, StructureAlgorithm, ParameterAlgorithm

try:
    learn('tests/synthetic_data/tsunami_subset_500.csv', StructureAlgorithm.GENETIC_K2, ParameterAlgorithm.MLE)
except Exception as e:
    traceback.print_exc()
