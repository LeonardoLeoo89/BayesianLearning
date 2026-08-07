import matplotlib.pyplot as plt
import networkx as nx
import pyagrum as gum
import os
from generate_networks import create_tsunami_network, create_allergy_network, create_train_delay_network

def draw_network(bn: gum.BayesNet, filename: str):
    """Draw a pyagrum BayesNet using NetworkX and Matplotlib."""
    G = nx.DiGraph()

    for i in bn.nodes():
        node_name = bn.variable(i).name()
        G.add_node(node_name)

    for u, v in bn.arcs():
        G.add_edge(bn.variable(u).name(), bn.variable(v).name())

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Seed for reproducible layout

    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=3000,
            edge_color='gray',
            linewidths=1,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20)

    plt.title(f"Bayesian Network: {bn.property('name') if 'name' in bn.properties() else 'Network'}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    os.makedirs("data/ground_truth", exist_ok=True)

    print("Generating networks...")
    tsunami_bn = create_tsunami_network()
    allergy_bn = create_allergy_network()
    train_bn = create_train_delay_network()

    print("Plotting networks...")
    draw_network(tsunami_bn, "data/ground_truth/tsunami.png")
    draw_network(allergy_bn, "data/ground_truth/allergy.png")
    draw_network(train_bn, "data/ground_truth/train_delay.png")

    print("\nSuccessfully generated plots!")
    print("You can view them by opening the PNG files in the 'data/ground_truth' directory.")
