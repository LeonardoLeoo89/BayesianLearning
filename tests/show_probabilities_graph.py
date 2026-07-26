import matplotlib.pyplot as plt
import networkx as nx
import pyagrum as gum
import os

def draw_network_with_probs(bn: gum.BayesNet, filename: str):
    """Draw a pyagrum BayesNet using NetworkX with marginal probabilities in the labels."""
    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    
    G = nx.DiGraph()
    labels = {}
    
    # Add nodes and construct labels with marginal probabilities
    for i in bn.nodes():
        node_name = bn.variable(i).name()
        G.add_node(node_name)
        
        # Calculate marginals
        posterior = ie.posterior(node_name)
        
        # Build the label text
        label_text = f"{node_name}\n" + "-"*len(node_name) + "\n"
        for idx in range(bn.variable(i).domainSize()):
            state = bn.variable(i).label(idx)
            prob = posterior[idx]
            label_text += f"{state}: {prob:.3f}\n"
            
        labels[node_name] = label_text.strip()
        
    # Add edges
    for u, v in bn.arcs():
        G.add_edge(bn.variable(u).name(), bn.variable(v).name())
        
    # Layout and plotting
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=1.5)  # Seed for reproducible layout, larger k for spacing
    
    # Draw
    nx.draw(G, pos, 
            labels=labels,
            node_color='lightgreen', 
            node_size=8000, 
            node_shape="s",  # square shape to better fit text
            edge_color='gray', 
            linewidths=1, 
            font_size=9, 
            font_weight='bold',
            font_family='monospace',
            arrows=True, 
            arrowsize=20)
            
    # Draw node borders manually since nx.draw square borders can sometimes be cut off
    ax = plt.gca()
    ax.margins(0.20)
            
    plt.title(f"Bayesian Network with Marginal Probabilities: {bn.property('name') if 'name' in bn.properties() else 'Network'}", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
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
