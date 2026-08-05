import os
import re
import pandas as pd
import numpy as np
import networkx as nx
import pyagrum as gum
import pyagrum.lib.bn_vs_bn as cmp
import pyagrum.lib.image as gumimage
import pydot
from PIL import Image, ImageDraw, ImageFont

def save_combined_diff_png(true_bn, pred_bn, output_path):
    # Generate temporary PNGs for each component at 150 DPI to prevent them from being too large
    true_dot = gumimage.BN2dot(true_bn)
    true_dot.set_dpi("150")
    true_dot.write_png("tmp_true.png")
    
    pred_dot = gumimage.BN2dot(pred_bn)
    pred_dot.set_dpi("150")
    pred_dot.write_png("tmp_pred.png")
    
    comparator = cmp.GraphicalBNComparator(true_bn, pred_bn)
    diff_dot = comparator.dotDiff()
    diff_dot.set_dpi("150")
    diff_dot.set_size("5") # Match original graph size limit
            
    diff_dot.write_png("tmp_diff.png")
    
    images = [Image.open(x).convert("RGBA") for x in ['tmp_true.png', 'tmp_pred.png', 'tmp_diff.png']]
    labels = ["Original Graph", "Inferred Graph", "Differences"]
    
    widths, heights = zip(*(i.size for i in images))
    
    padding = 60
    max_width = max(widths) + padding * 2
    total_height = sum(heights) + padding * 4 + 150 # extra 150 for text labels
    
    # White background instead of transparent
    new_im = Image.new('RGBA', (max_width, total_height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(new_im)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35)
    except:
        font = ImageFont.load_default()
    
    y_offset = padding
    for i, (img, label) in enumerate(zip(images, labels)):
        text_color = "black"
        if label == "Differences":
            # Draw a dark background for the differences section and down
            draw.rectangle([0, y_offset - padding//2, max_width, total_height], fill="#202020")
            text_color = "white"
            
        # Center the text
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except:
            text_width = len(label) * 6 # fallback approximation
            
        x_text = (max_width - text_width) // 2
        draw.text((x_text, y_offset), label, fill=text_color, font=font)
        y_offset += 50
        
        # Center the image
        x_img = (max_width - img.width) // 2
        new_im.paste(img, (x_img, y_offset), mask=img)
        y_offset += img.height + padding
        
    new_im.save(output_path)
    
    # Cleanup temps
    for tmp in ['tmp_true.png', 'tmp_pred.png', 'tmp_diff.png']:
        if os.path.exists(tmp):
            os.remove(tmp)

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



def make_dag(pred_edges, nodes):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    edge_counts = {}
    for u, v in pred_edges:
        edge_counts[(u, v)] = 1
        
    directed = []
    undirected = []
    for u, v in pred_edges:
        if (v, u) in edge_counts:
            if (u, v) not in undirected and (v, u) not in undirected:
                undirected.append((u, v))
        else:
            directed.append((u, v))
            
    for u, v in directed:
        G.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
            
    for u, v in undirected:
        G.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
            G.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(v, u)
                
    return list(G.edges())

def sort_bn(bn: gum.BayesNet) -> gum.BayesNet:
    sorted_bn = gum.BayesNet()
    for n in bn.names():
        var = bn.variableFromName(n)
        labels = sorted([var.label(i) for i in range(var.domainSize())])
        new_var = gum.LabelizedVariable(n, n, 0)
        for lbl in labels: new_var.addLabel(lbl)
        sorted_bn.add(new_var)
        
    for u, v in bn.arcs():
        sorted_bn.addArc(bn.variable(u).name(), bn.variable(v).name())
        
    for n in bn.names():
        cpt, new_cpt = bn.cpt(n), sorted_bn.cpt(n)
        inst_new, inst_old = gum.Instantiation(new_cpt), gum.Instantiation(cpt)
        inst_new.setFirst()
        while not inst_new.end():
            for var_name in new_cpt.names:
                val_str = sorted_bn.variableFromName(var_name).label(inst_new.val(var_name))
                inst_old.chgVal(var_name, val_str)
            new_cpt.set(inst_new, cpt.get(inst_old))
            inst_new.inc()
    return sorted_bn

def calculate_kl_divergence(true_bn: gum.BayesNet, pred_edges: list, dataset_path: str) -> dict:
    from bayesian_learning.categorical_model.learn_categorical import learn_parameters, ParameterAlgorithm
    import pandas as pd
    try:
        df = pd.read_csv(dataset_path)
        nodes = list(df.columns)
        
        sorted_true_bn = sort_bn(true_bn)
        pred_bn_aligned = gum.BayesNet()
        
        for n in nodes:
            pred_bn_aligned.add(sorted_true_bn.variableFromName(n))
            
        dag_edges = make_dag(pred_edges, set(nodes))
        for u, v in dag_edges:
            pred_bn_aligned.addArc(u, v)
        
        pred_bn_fitted = learn_parameters(dataset_path, pred_bn_aligned.dag(), ParameterAlgorithm.MLE)
        
        dist = gum.ExactBNdistance(sorted_true_bn, pred_bn_fitted)
        res = dist.compute()
        
        return {
            'KL_Div': res.get('klPQ', float('nan')),
            'Hellinger': res.get('hellinger', float('nan')),
            'Bhattacharyya': res.get('bhattacharya', float('nan')),
            'Jensen_Shannon': res.get('jensen-shannon', float('nan'))
        }
    except Exception as e:
        print(f"Error calculating KL divergence: {e}")
        return {
            'KL_Div': float('nan'),
            'Hellinger': float('nan'),
            'Bhattacharyya': float('nan'),
            'Jensen_Shannon': float('nan')
        }

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
    files = [f for f in os.listdir(input_dir) if f.endswith('.bif') or f.endswith('.csv')]
    
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
                
                # Plot graphical diff if we have the true BN
                true_bif_path = f"generated_bns/{ds_name}.bif"
                if os.path.exists(true_bif_path):
                    try:
                        true_bn = gum.loadBN(true_bif_path)
                        diff_path = os.path.join(input_dir, f"{f}_diff.png")
                        save_combined_diff_png(true_bn, bn, diff_path)
                    except Exception as e:
                        print(f"Failed to plot diff for {f}: {e}")
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
                
            metrics = calculate_metrics(true_edges, pred_edges, nodes)
            
            # Compute KL Divergence if it's a categorical dataset and we have the true BN
            if 'sem' not in ds_name:
                true_bif_path = f"generated_bns/{ds_name}.bif"
                dataset_path = os.path.join("tests", "synthetic_data", f"{ds_name}_samples_subset_500.csv")
                if os.path.exists(true_bif_path) and os.path.exists(dataset_path):
                    true_bn = gum.loadBN(true_bif_path)
                    distances = calculate_kl_divergence(true_bn, pred_edges, dataset_path)
                    metrics.update(distances)
                else:
                    metrics.update({'KL_Div': float('nan'), 'Hellinger': float('nan'), 'Bhattacharyya': float('nan'), 'Jensen_Shannon': float('nan')})
            else:
                metrics.update({'KL_Div': float('nan'), 'Hellinger': float('nan'), 'Bhattacharyya': float('nan'), 'Jensen_Shannon': float('nan')})
            
            metrics['Dataset'] = ds_name
            metrics['Algorithm'] = algo
            metrics['Algorithm_Type'] = "SEM" if "sem" in ds_name else "Categorical"
            results.append(metrics)
        except Exception as e:
            print(f"Failed on {f}: {e}")
            
    # Print Markdown Table
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No results found.")
        return
        
    df_res = df_res.sort_values(by=["Algorithm_Type", "Dataset", "Algorithm"])
    df_res = df_res[["Algorithm_Type", "Dataset", "Algorithm", "SHD", "TP", "FP", "FN", "F1", "KL_Div", "Hellinger", "Bhattacharyya", "Jensen_Shannon"]]
    
    md_table = df_res.to_markdown(index=False)
    print(md_table)
    
    with open("tests/benchmark_results/accuracy_report.md", "w") as out:
        out.write("# Structural Accuracy Report\n\n")
        out.write("This table compares the learned networks against the true ground-truth network structures that generated the data.\n\n")
        out.write("- **SHD**: Structural Hamming Distance (lower is better, 0 is perfect). The number of edge additions, deletions, or reversals needed to match the true graph.\n")
        out.write("- **TP**: True Positives (correct edges)\n")
        out.write("- **FP**: False Positives (extra edges)\n")
        out.write("- **FN**: False Negatives (missing edges)\n")
        out.write("- **F1**: F1 Score (higher is better, 1.0 is perfect)\n")
        out.write("- **KL_Div**: Kullback-Leibler Divergence (lower is better, 0 is perfect). Measures the statistical information lost due to structural errors by applying parameter learning to both the true and inferred structures.\n\n")
        out.write(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
