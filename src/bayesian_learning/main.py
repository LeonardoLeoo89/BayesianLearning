import numpy as np
import pandas as pd
import pyagrum as gum

def main():
    print("Bayesian Learning Python Project Initialized!")
    
    # Simple pyagrum test
    bn = gum.BayesNet('WaterSprinkler')
    print("Created BayesNet:", bn)

if __name__ == "__main__":
    main()
