import os
import re
import pandas as pd
import numpy as np
import networkx as nx
import pyagrum as gum

# Define true networks for SEM datasets
TRUE_SEM_EDGES = {
    'tsunami_sem': [
        ('Earthquake', 'TsunamiHeight'), ('SubmarineProximity', 'TsunamiHeight'),
        ('Earthquake', 'WarningUrgency'), ('TsunamiHeight', 'WarningUrgency')
    ],
    'allergy_sem': [
        ('Atopy', 'DustMiteIgE'), ('Atopy', 'PollenIgE'),
        ('PollenIgE', 'BirchPollenIgE'), ('PollenIgE', 'RhinitisSev'),
        ('DustMiteIgE', 'AsthmaSev'), ('BirchPollenIgE', 'AppleIgE'),
        ('BirchPollenIgE', 'HazelnutIgE')
    ],
    'train_delay_sem': [
        ('SeasonalFactor', 'WeatherSev'), ('TimeOfDayRush', 'PassengerVol'),
        ('WeatherSev', 'TrackIncident'), ('WeatherSev', 'InfraFailure'),
        ('TrackIncident', 'InfraFailure'), ('TrackIncident', 'SpeedRestriction'),
        ('WeatherSev', 'SpeedRestriction'), ('PassengerVol', 'HubCongestion'),
        ('InfraFailure', 'DepartureDelay'), ('HubCongestion', 'DepartureDelay'),
        ('DepartureDelay', 'ArrivalDelay'), ('SpeedRestriction', 'ArrivalDelay'),
        ('ArrivalDelay', 'CompensationClaim')
    ]
}

def load_true_categorical_edges(name: str) -> list:
    bif_path = f"generated_bns/{name}.bif"
    if not os.path.exists(bif_path): return []
    bn = gum.loadBN(bif_path)
    edges = []
    for u, v in bn.arcs():
        edges.append((bn.variable(u).name(), bn.variable(v).name()))
    return edges

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
                else: # ---
                    G.add_edge(u, v)
                    G.add_edge(v, u) # Undirected penalizes as extra edges in strict SHD
    return G

def calculate_metrics(true_edges, pred_edges, nodes):
    # Convert to sets of tuples for easier math
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    
    # Structural Hamming Distance
    # Reverse edges
    reversed_edges = 0
    for u, v in list(pred_set):
        if (v, u) in true_set and (u, v) not in true_set:
            reversed_edges += 1
            pred_set.remove((u, v))
            pred_set.add((v, u))
            
    # After flipping reversed edges, compute TP, FP, FN
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    shd = reversed_edges + fp + fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "SHD": shd,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F1": f1
    }

def main():
    input_dir = "tests/benchmark_results"
    files = [f for f in os.listdir(input_dir) if f.endswith('.bif') or f.endswith('.csv') or f.endswith('.txt')]
    
    results = []
    
    for f in files:
        if f == "benchmark_results.csv": continue
        filepath = os.path.join(input_dir, f)
        
        # Determine dataset name and algorithm
        base_name = f.replace("_subset_500.csv", "")
        if "adjacency" in base_name:
            ds_name = base_name.split("_sem_")[0] + "_sem"
            algo = base_name.split("_sem_")[1].replace("_adjacency.csv", "")
        else:
            ds_name = base_name.split("_samples_")[0]
            algo = base_name.split("_samples_")[1]
            if algo.endswith(".bif"): algo = algo[:-4]
            if algo.endswith(".txt"): algo = algo[:-4]
            
        # Get True Edges
        if ds_name in TRUE_SEM_EDGES:
            true_edges = TRUE_SEM_EDGES[ds_name]
        else:
            true_edges = load_true_categorical_edges(ds_name)
            if not true_edges:
                continue
                
        # Get nodes from true edges to be safe
        nodes = set([u for u, v in true_edges] + [v for u, v in true_edges])
        
        # Load Pred Edges
        pred_edges = []
        try:
            if f.endswith('.bif'):
                bn = gum.loadBN(filepath)
                for u, v in bn.arcs():
                    pred_edges.append((bn.variable(u).name(), bn.variable(v).name()))
            elif f.endswith('.csv'):
                df = pd.read_csv(filepath)
                mat = df.values
                if df.columns[0] == '0' or df.columns[0] == 0:
                    orig_df = pd.read_csv(f"tests/sem_data/{ds_name}.csv")
                    node_names = orig_df.columns.tolist()
                else:
                    node_names = df.columns.tolist()
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if abs(mat[i, j]) > 0.1:
                            pred_edges.append((node_names[i], node_names[j]))
            elif f.endswith('.txt'):
                G = parse_tetrad_txt(filepath)
                pred_edges = list(G.edges())
                
            metrics = calculate_metrics(true_edges, pred_edges, nodes)
            metrics['Dataset'] = ds_name
            metrics['Algorithm'] = algo
            metrics['Algorithm_Type'] = "SEM" if "sem" in ds_name else "Categorical"
            results.append(metrics)
        except Exception as e:
            pass # Failed to load
            
    # Print Markdown Table
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No results found.")
        return
        
    df_res = df_res.sort_values(by=["Algorithm_Type", "Dataset", "Algorithm"])
    df_res = df_res[["Algorithm_Type", "Dataset", "Algorithm", "SHD", "TP", "FP", "FN", "F1"]]
    
    print(df_res.to_markdown(index=False))
    
    with open("tests/benchmark_results/accuracy_report.md", "w") as out:
        out.write("# Structural Accuracy Report\n\n")
        out.write("This table compares the learned networks against the true ground-truth network structures that generated the data.\n\n")
        out.write("- **SHD**: Structural Hamming Distance (lower is better, 0 is perfect). The number of edge additions, deletions, or reversals needed to match the true graph.\n")
        out.write("- **TP**: True Positives (correct edges)\n")
        out.write("- **FP**: False Positives (extra edges)\n")
        out.write("- **FN**: False Negatives (missing edges)\n")
        out.write("- **F1**: F1 Score (higher is better, 1.0 is perfect)\n\n")
        out.write(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
