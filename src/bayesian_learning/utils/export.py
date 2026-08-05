import os
import networkx as nx
import matplotlib.pyplot as plt
import pyagrum as gum
import pandas as pd
from bayesian_learning.SEM_model.result import SEMResult

def plot_and_save_networkx(G: nx.DiGraph, output_path: str):
    """Utility function to plot a networkx DiGraph and save it to a file."""
    plt.figure(figsize=(10, 8))
    # Use spring layout or kamada_kawai for nice visualization
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)
        
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold', 
            arrows=True, arrowsize=20)
    plt.title("Learned Directed Acyclic Graph")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def export_categorical_results(bn: gum.BayesNet, output_prefix: str):
    """
    Exports a Categorical BayesNet.
    - Saves the native BIF format
    - Saves a graphical plot (PNG)
    - Saves the CPTs to a readable text file
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    
    # 1. Save native BIF
    bif_path = f"{output_prefix}.bif"
    gum.saveBN(bn, bif_path)
    
    # 2. Extract and Plot Graph
    G = nx.DiGraph()
    for node_id in bn.nodes():
        G.add_node(bn.variable(node_id).name())
    for source, target in bn.arcs():
        G.add_edge(bn.variable(source).name(), bn.variable(target).name())
        
    plot_path = f"{output_prefix}_graph.png"
    plot_and_save_networkx(G, plot_path)
    
    # 3. Save CPTs
    cpt_path = f"{output_prefix}_cpts.txt"
    with open(cpt_path, "w") as f:
        for node_id in bn.nodes():
            name = bn.variable(node_id).name()
            f.write(f"--- CPT for {name} ---\n")
            f.write(str(bn.cpt(node_id)))
            f.write("\n\n")
            
    return bif_path, plot_path, cpt_path

def export_categorical_from_bif(bif_file: str, output_prefix: str):
    """Helper method to load a BIF file and export its graphics and CPTs."""
    bn = gum.loadBN(bif_file)
    return export_categorical_results(bn, output_prefix)

def export_sem_results(result: SEMResult, output_prefix: str):
    """
    Exports a SEM Result.
    - Saves the Adjacency Matrix to CSV
    - Saves a graphical plot (PNG)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    
    # 1. Save Adjacency CSV
    adj_path = f"{output_prefix}_adjacency.csv"
    pd.DataFrame(result.adjacency_matrix).to_csv(adj_path, index=False)
    
    # 2. Extract and Plot Graph
    G = result.to_networkx()
    plot_path = f"{output_prefix}_graph.png"
    plot_and_save_networkx(G, plot_path)
    
    return adj_path, plot_path
