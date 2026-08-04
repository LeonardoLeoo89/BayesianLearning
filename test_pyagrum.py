import pyagrum as gum
import pandas as pd
import numpy as np

# create a dummy dataset
df = pd.DataFrame(np.random.randint(0,2,size=(100, 4)), columns=list('ABCD'))
df.to_csv('dummy.csv', index=False)

learner = gum.BNLearner('dummy.csv')
learner.useK2([0,1,2,3])
bn = learner.learnBN()
nodes_list = list(bn.nodes())
scores = [learner.score(node) for node in nodes_list]
print("Scores:", scores)
print("Sum:", sum(scores))
