import pydot
import pyagrum as gum
import os

def draw_network_with_probs(bn: gum.BayesNet, filename: str):
    """Draw a pyagrum BayesNet using PyDot."""
    graph = pydot.Dot(graph_type="digraph", bgcolor="white", rankdir="TB")
    graph.set_node_defaults(shape="box", style="rounded,filled", fillcolor="lightgreen", fontname="monospace", fontsize="10")
    graph.set_edge_defaults(color="gray40", arrowhead="normal", arrowsize="1.0", penwidth="1.2")

    for i in bn.nodes():
        node_name = bn.variable(i).name()
        node = pydot.Node(node_name, label=node_name)
        graph.add_node(node)
        
    for u, v in bn.arcs():
        u_name = bn.variable(u).name()
        v_name = bn.variable(v).name()
        edge = pydot.Edge(u_name, v_name)
        graph.add_edge(edge)
        
    graph.write_png(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    os.makedirs("tests/probability_plots", exist_ok=True)
    
    # Load networks
    tsunami_bn = gum.loadBN("generated_bns/tsunami.bif")
    allergy_bn = gum.loadBN("generated_bns/allergy.bif")
    train_bn = gum.loadBN("generated_bns/train_delay.bif")
    
    print("Generating network graphs with inline probabilities...")
    draw_network_with_probs(tsunami_bn, "tests/probability_plots/tsunami_graph_probs.png")
    draw_network_with_probs(allergy_bn, "tests/probability_plots/allergy_graph_probs.png")
    draw_network_with_probs(train_bn, "tests/probability_plots/train_delay_graph_probs.png")
    
    print("\nSuccessfully generated plots!")
