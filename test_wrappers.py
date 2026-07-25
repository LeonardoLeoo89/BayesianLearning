import pandas as pd
import numpy as np
from bayesian_learning.SEM_model.wrappers.gran_dag_wrapper import GraNDAGWrapper
from bayesian_learning.SEM_model.wrappers.dag_gnn_wrapper import DAGGNNWrapper

data = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])

print("Testing GraN-DAG...")
gran_wrapper = GraNDAGWrapper(num_train_iter=10)
gran_res = gran_wrapper.learn(data)
print("GraN-DAG graph:", gran_res.adjacency_matrix)
print("Testing GraN-DAG predict_distribution...")
dists = gran_res.predict_distribution(data.values[:2]) # Test on first two samples
print("Predicted distributions length:", len(dists))
for i, dist in enumerate(dists):
    print(f"Var {i} distribution type:", type(dist))

print("Testing DAG-GNN...")
dag_wrapper = DAGGNNWrapper(epochs=2)
dag_res = dag_wrapper.learn(data)
print("DAG-GNN graph:", dag_res.adjacency_matrix)
print("Done.")
