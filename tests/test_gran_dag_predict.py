import numpy as np
import pandas as pd
from bayesian_learning.SEM_model.learn_sem import learn_structure, SEMAlgorithm
from bayesian_learning.SEM_model.result import GranDAGResult

def main():
    df = pd.read_csv("tests/sem_data/allergy_sem_subset_500_std.csv")
    
    # Train GraN-DAG quickly
    result = learn_structure(
        "tests/sem_data/allergy_sem_subset_500_std.csv", 
        SEMAlgorithm.GRAN_DAG,
        epochs=1,
        iterations=10,
        batch_size=64
    )
    
    if not isinstance(result, GranDAGResult):
        print("Not a GranDAGResult")
        return
        
    print("Model trained.")
    
    # Take first 5 samples
    X_test = df.values[:5]
    
    # Test predict_distribution
    distributions = result.predict_distribution(X_test)
    print("Distributions predicted successfully.")
    print(f"Number of distributions (one per var): {len(distributions)}")
    for i, dist in enumerate(distributions):
        print(f"Var {i}: {type(dist)}")

if __name__ == "__main__":
    main()
