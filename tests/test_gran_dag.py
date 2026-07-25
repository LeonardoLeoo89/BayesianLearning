import pandas as pd
import numpy as np
import sys
import os

# Add src directory to PYTHONPATH for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bayesian_learning.SEM_model.wrappers.gran_dag_wrapper import GraNDAGWrapper

def test_gran_dag():
    print("Generating synthetic data...")
    # Generate simple synthetic data: X -> Y -> Z
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.1
    z = -1.5 * y + np.random.randn(n) * 0.1
    
    data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    
    # We use a very small number of iterations (10) for testing to ensure it runs quickly.
    # GraN-DAG defaults to 100,000 iterations for convergence, but we just want to verify the pipeline.
    wrapper = GraNDAGWrapper(num_train_iter=10)
    
    print("Testing GraN-DAG Wrapper...")
    try:
        result = wrapper.learn(data)
        print("GraN-DAG ran successfully!")
        print(f"Output Graph Shape: {result.adjacency_matrix.shape}")
        print("Output Graph:")
        print(result.adjacency_matrix)
    except Exception as e:
        print("Failed to run GraN-DAG:")
        import traceback
        traceback.print_exc()
        raise e
        
    print("Testing GraN-DAG predict_distribution...")
    distributions = result.predict_distribution(data.values[:5])
    print(f"Returned {len(distributions)} distributions!")
    for i, dist in enumerate(distributions):
        print(f"Node {i} Distribution Type: {type(dist)}")
        print(f"Node {i} Mean Sample: {dist.mean}")
        
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_gran_dag()
