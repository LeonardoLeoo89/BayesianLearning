import os
import networkx as nx
import matplotlib.pyplot as plt
import pyagrum as gum
import pandas as pd
from bayesian_learning.SEM_model.result import SEMResult

def plot_and_save_networkx(G: nx.DiGraph, output_path: str):
    """Utility function to plot a networkx DiGraph and save it to a file using PyDot."""
    import pydot
    graph = pydot.Dot(graph_type="digraph", bgcolor="white", rankdir="TB")
    graph.set_node_defaults(shape="box", style="rounded,filled", fillcolor="lightgreen", fontname="monospace", fontsize="10")
    graph.set_edge_defaults(color="gray40", arrowhead="normal", arrowsize="1.0", penwidth="1.2")
    
    for n in G.nodes():
        node = pydot.Node(str(n), label=str(n))
        graph.add_node(node)
        
    for u, v in G.edges():
        edge = pydot.Edge(str(u), str(v))
        graph.add_edge(edge)
        
    graph.write_png(output_path)

def export_categorical_results(bn: gum.BayesNet, output_prefix: str):
    """
    Exports a Categorical BayesNet.
    - Saves the native BIF format
    - Saves a graphical plot (PNG)
    - Saves the CPTs to a readable text file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)

    bif_path = f"{output_prefix}.bif"
    gum.saveBN(bn, bif_path)

    import pyagrum.lib.image as gumimage
    dot_obj = gumimage.BN2dot(bn)
    dot_obj.set_dpi("300")
    plot_path = f"{output_prefix}_graph.png"
    dot_obj.write_png(plot_path)

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

    adj_path = f"{output_prefix}_adjacency.csv"
    pd.DataFrame(result.adjacency_matrix).to_csv(adj_path, index=False)

    G = result.to_networkx()
    plot_path = f"{output_prefix}_graph.png"
    plot_and_save_networkx(G, plot_path)

    return adj_path, plot_path
