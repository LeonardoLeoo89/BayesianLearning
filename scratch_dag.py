import pyagrum as gum
import pandas as pd
import os

# Create a dummy CSV
df = pd.DataFrame({
    'A': [0, 1, 0, 1],
    'B': [1, 1, 0, 0]
})
df.to_csv('dummy.csv', index=False)

learner = gum.BNLearner('dummy.csv')

# Create a DAG with node IDs
dag = gum.DAG()
# We must add nodes first. Let's see if we add node 0 and 1
n0 = dag.addNode() # returns 0
n1 = dag.addNode() # returns 1
dag.addArc(n0, n1)

print("DAG nodes:", dag.nodes())

# Learn parameters
bn = learner.learnParameters(dag)
print("BN nodes:", bn.names())
print("BN CPT for B:")
print(bn.cpt(n1))

os.remove('dummy.csv')
