import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bayesian_learning.categorical_model.learn_categorical import learn, StructureAlgorithm, ParameterAlgorithm

print("Testing Genetic K2...")
bn_k2 = learn('tests/synthetic_data/allergy_samples_subset_500.csv', StructureAlgorithm.GENETIC_K2, ParameterAlgorithm.MLE)
print(f"Nodes: {bn_k2.names()}")
print(f"Arcs: {bn_k2.arcs()}")
print("Testing Hill Climbing...")
bn_hc = learn('tests/synthetic_data/allergy_samples_subset_500.csv', StructureAlgorithm.HILL_CLIMBING, ParameterAlgorithm.MLE)
print(f"Nodes: {bn_hc.names()}")
print(f"Arcs: {bn_hc.arcs()}")
print("Testing PC...")
bn_pc = learn('tests/synthetic_data/allergy_samples_subset_500.csv', StructureAlgorithm.PC, ParameterAlgorithm.MLE)
print(f"Nodes: {bn_pc.names()}")
print(f"Arcs: {bn_pc.arcs()}")
print("Testing FCI...")
bn_fci = learn('tests/synthetic_data/allergy_samples_subset_500.csv', StructureAlgorithm.FCI, ParameterAlgorithm.MLE)
print(f"Nodes: {bn_fci.names()}")
print(f"Arcs: {bn_fci.arcs()}")
print(f"Nodes: {bn_hc_shrink.names()}")
print(f"Arcs: {bn_hc_shrink.arcs()}")
print("Done!")
