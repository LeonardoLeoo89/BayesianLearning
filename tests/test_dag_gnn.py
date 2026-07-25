import pandas as pd
import numpy as np
from bayesian_learning.SEM_model.learn_sem import SEMAlgorithm, learn

def test_dag_gnn():
    print("Generating synthetic data...")
    # Generate some dummy data (100 samples, 5 variables)
    data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    
    print("Testing DAG-GNN Wrapper...")
    # We pass a tiny number of epochs just to ensure it runs and finishes quickly without crashing
    data_path = "dummy.csv"
    data.to_csv(data_path, index=False)
    
    # Run the wrapper
    # Note: learn() takes a string path, not the dataframe directly.
    # We will pass epochs=1 just to test if the pipeline runs successfully
    from bayesian_learning.SEM_model.wrappers.dag_gnn_wrapper import DAGGNNWrapper
    wrapper = DAGGNNWrapper(epochs=1)
    
    try:
        result = wrapper.learn(data)
        print("DAG-GNN ran successfully!")
        print("Output Graph Shape:", result.adjacency_matrix.shape)
        print("Output Graph:\n", result.adjacency_matrix)
    except Exception as e:
        print("Failed to run DAG-GNN:")
        raise e

if __name__ == "__main__":
    test_dag_gnn()
