import os
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pyagrum as gum

def plot_graph(G: nx.DiGraph, title: str, output_path: str):
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, k=1.5, seed=42)

    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=3000,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            arrowsize=20)

    plt.title(title, fontsize=14, pad=20)
    plt.margins(0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def parse_tetrad_txt(filepath: str) -> nx.DiGraph:
    G = nx.DiGraph()

    with open(filepath, 'r') as f:
        content = f.read()

    edges_section = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Graph Edges:'):
            edges_section = True
            continue
        elif edges_section and line == '':
            edges_section = False
            continue
        elif edges_section and line.startswith('Graph Attributes:'):
            break

        if edges_section:
            match = re.search(r'\d+\.\s+(\w+)\s+(.+)\s+(\w+)', line)
            if match:
                u, edge_type, v = match.groups()
                G.add_node(u)
                G.add_node(v)

                if '-->' in edge_type or 'o->' in edge_type:
                    G.add_edge(u, v)
                elif '<--' in edge_type or '<-o' in edge_type:
                    G.add_edge(v, u)
                elif '<->' in edge_type or 'o-o' in edge_type:
                    G.add_edge(u, v)
                    G.add_edge(v, u)
                else:
                    G.add_edge(u, v)

    return G

def plot_all_benchmark_graphs(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for f in files:
        filepath = os.path.join(input_dir, f)
        base_name = os.path.splitext(f)[0]
        output_path = os.path.join(output_dir, f"{base_name}_graph.png")

        if f.endswith('.bif'):
            try:
                bn = gum.loadBN(filepath)
                G = nx.DiGraph()
                for i in bn.nodes():
                    G.add_node(bn.variable(i).name())
                for u, v in bn.arcs():
                    G.add_edge(bn.variable(u).name(), bn.variable(v).name())

                title = f.replace('_subset_500.csv', '').replace('.bif', '').replace('_', ' ')
                plot_graph(G, title, output_path)
                print(f"Generated {output_path}")
            except Exception as e:
                print(f"Error plotting {f}: {e}")

        elif f.endswith('_adjacency.csv'):
            try:
                df = pd.read_csv(filepath)
                mat = df.values

                if df.columns[0] == '0' or df.columns[0] == 0:
                    dataset_name = f.split('_sem')[0] + "_sem_subset_500.csv"
                    dataset_path = os.path.join(input_dir.replace("benchmark_results", "sem_data"), dataset_name)
                    if os.path.exists(dataset_path):
                        orig_df = pd.read_csv(dataset_path)
                        node_names = orig_df.columns.tolist()
                    else:
                        node_names = [f"Node_{i}" for i in range(mat.shape[0])]
                else:
                    node_names = df.columns.tolist()

                G = nx.DiGraph()
                for i in range(mat.shape[0]):
                    G.add_node(node_names[i])

                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if abs(mat[i, j]) > 0.1:
                            G.add_edge(node_names[i], node_names[j])

                title = f.replace('_subset_500.csv', '').replace('_adjacency.csv', '').replace('_', ' ')
                plot_graph(G, title, output_path)
                print(f"Generated {output_path}")
            except Exception as e:
                print(f"Error plotting {f}: {e}")

        elif f.endswith('.txt') and "(Tetrad)" in f:
            try:
                G = parse_tetrad_txt(filepath)
                if len(G.nodes) > 0:
                    title = f.replace('_subset_500.csv', '').replace('.txt', '').replace('_', ' ')
                    plot_graph(G, title, output_path)
                    print(f"Generated {output_path}")
            except Exception as e:
                print(f"Error plotting {f}: {e}")

if __name__ == "__main__":
    plot_all_benchmark_graphs("results/benchmarks", "results/benchmark_graphs")
    print("Finished generating plots!")
