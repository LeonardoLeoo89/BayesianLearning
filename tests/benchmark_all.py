import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyagrum as gum
from typing import Dict, Any, List

from bayesian_learning.categorical_model.learn_categorical import (
    StructureAlgorithm, ParameterAlgorithm, learn
)
from bayesian_learning.SEM_model.learn_sem import (
    SEMAlgorithm, learn_structure
)

# Paths to datasets
CAT_DATA_DIR = "tests/synthetic_data"
SEM_DATA_DIR = "tests/sem_data"
OUTPUT_DIR = "tests/benchmark_results"

# Max samples constraint from user
MAX_SAMPLES = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

def subset_data(filepath: str, max_samples: int) -> str:
    """Creates a temporary subset of the dataset and returns the path."""
    df = pd.read_csv(filepath)
    if len(df) > max_samples:
        df = df.head(max_samples)
    
    subset_path = filepath.replace(".csv", f"_subset_{max_samples}.csv")
    df.to_csv(subset_path, index=False)
    return subset_path

def standardize_sem_data(filepath: str) -> str:
    """Standardizes continuous features to zero mean and unit variance."""
    df = pd.read_csv(filepath)
    df = (df - df.mean()) / df.std()
    std_path = filepath.replace(".csv", "_std.csv")
    df.to_csv(std_path, index=False)
    return std_path

def mask_data_generically(filepath: str, mask_frac: float = 0.3) -> str:
    """Masks 30% of the data in two random columns to test Structural EM."""
    df = pd.read_csv(filepath)
    np.random.seed(42)
    cols_to_mask = np.random.choice(df.columns, size=min(2, len(df.columns)), replace=False)
    for col in cols_to_mask:
        df[col] = df[col].astype(str)
        mask = np.random.rand(len(df)) < mask_frac
        df.loc[mask, col] = '?'
        
    masked_path = filepath.replace('.csv', '_masked.csv')
    df.to_csv(masked_path, index=False)
    return masked_path

def benchmark_categorical(filepath: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    
    algorithms = {
        "Hill Climbing (Agrum)": (learn, [StructureAlgorithm.HILL_CLIMBING, ParameterAlgorithm.MLE]),
        "Genetic K2": (learn, [StructureAlgorithm.GENETIC_K2, ParameterAlgorithm.MLE]),
        "PC (Tetrad)": (learn, [StructureAlgorithm.PC, ParameterAlgorithm.MLE]),
        "FCI (Tetrad)": (learn, [StructureAlgorithm.FCI, ParameterAlgorithm.MLE]),
        "RFCI (Tetrad)": (learn, [StructureAlgorithm.RFCI, ParameterAlgorithm.MLE]),
        "Structural EM": (learn, [StructureAlgorithm.STRUCTURAL_EM, ParameterAlgorithm.EM])
    }
    
    for name, (func, args) in algorithms.items():
        print(f"  Running {name}...")
        start_time = time.perf_counter()
        
        output = None
        try:
            # Mask data specifically for Structural EM
            target_filepath = filepath
            if name == "Structural EM":
                target_filepath = mask_data_generically(filepath, 0.3)
                
            # Tetrad wrappers just return Java Graph, Agrum wrappers return gum.BayesNet
            output = func(target_filepath, *args)
            
            if isinstance(output, gum.BayesNet):
                edges = output.sizeArcs()
                # Save graph/parameters
                out_bif = os.path.join(OUTPUT_DIR, f"{os.path.basename(filepath)}_{name.replace(' ', '_')}.bif")
                gum.saveBN(output, out_bif)
            else:
                # Tetrad Java Graph (we just log it worked)
                edges = len(output.getEdges())
                # Note: We can't easily save Tetrad Java graphs natively in pyagrum format, 
                # but we'll print its string representation.
                out_txt = os.path.join(OUTPUT_DIR, f"{os.path.basename(filepath)}_{name.replace(' ', '_')}.txt")
                with open(out_txt, "w") as f:
                    f.write(str(output))
                    
            status = "Success"
        except Exception as e:
            import traceback
            traceback.print_exc()
            edges = 0
            status = f"Failed: {str(e)}"
            
        elapsed = time.perf_counter() - start_time
        
        results[name] = {
            "time": elapsed,
            "edges": edges,
            "status": status
        }
        print(f"    -> Time: {elapsed:.3f}s, Status: {status}, Edges: {edges}")
        
    return results

def benchmark_sem(filepath: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    
    algorithms = {
        "DAGMA": (SEMAlgorithm.DAGMA, {}),
        "DAG_GNN": (SEMAlgorithm.DAG_GNN, {}),
        "GraN-DAG": (SEMAlgorithm.GRAN_DAG, {})
    }
    
    for name, (algo_enum, kwargs) in algorithms.items():
        print(f"  Running {name}...")
        start_time = time.perf_counter()
        
        output = None
        try:
            sem_result = learn_structure(filepath, algo_enum, **kwargs)
            adjacency_matrix = sem_result.adjacency_matrix
            edges = int(np.sum(adjacency_matrix != 0))
            status = "Success"
            
            # Save the adjacency matrix
            out_csv = os.path.join(OUTPUT_DIR, f"{os.path.basename(filepath)}_{name}_adjacency.csv")
            pd.DataFrame(adjacency_matrix).to_csv(out_csv, index=False)
            
        except Exception as e:
            edges = 0
            status = f"Failed: {str(e)}"
            
        elapsed = time.perf_counter() - start_time
        
        results[name] = {
            "time": elapsed,
            "edges": edges,
            "status": status
        }
        print(f"    -> Time: {elapsed:.3f}s, Status: {status}, Edges: {edges}")
        
    return results

def plot_results(all_results: Dict[str, Dict[str, Any]], title: str, output_img: str):
    # all_results = { dataset_name: { algo_name: {time, edges, status} } }
    datasets = list(all_results.keys())
    
    # Collect all algorithms present
    algorithms = set()
    for res in all_results.values():
        algorithms.update(res.keys())
    algorithms = list(algorithms)
    
    # Prepare data for plotting
    times = {algo: [] for algo in algorithms}
    
    for ds in datasets:
        for algo in algorithms:
            time_val = all_results[ds].get(algo, {}).get("time", 0)
            status = all_results[ds].get(algo, {}).get("status", "Missing")
            if not status.startswith("Success"):
                time_val = 0 # or plot differently
            times[algo].append(time_val)
            
    x = np.arange(len(datasets))
    width = 0.8 / len(algorithms)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        ax.bar(x + i * width - 0.4 + width/2, times[algo], width, label=algo)
        
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()

if __name__ == "__main__":
    print(f"=== Starting Benchmark (Max Samples: {MAX_SAMPLES}) ===")
    
    cat_results = {}
    sem_results = {}
    
    # 1. Benchmark Categorical
    print("\n--- Benchmarking Categorical Models ---")
    cat_files = [f for f in os.listdir(CAT_DATA_DIR) if f.endswith(".csv") and "subset" not in f]
    for f in cat_files:
        filepath = os.path.join(CAT_DATA_DIR, f)
        print(f"Dataset: {f}")
        subset_path = subset_data(filepath, MAX_SAMPLES)
        cat_results[f] = benchmark_categorical(subset_path)
        
    # 2. Benchmark SEM
    print("\n--- Benchmarking SEM Models ---")
    sem_files = [f for f in os.listdir(SEM_DATA_DIR) if f.endswith(".csv") and "subset" not in f]
    for f in sem_files:
        filepath = os.path.join(SEM_DATA_DIR, f)
        print(f"Dataset: {f}")
        subset_path = subset_data(filepath, MAX_SAMPLES)
        std_path = standardize_sem_data(subset_path)
        sem_results[f] = benchmark_sem(std_path)
        
    # 3. Export raw data to CSV
    print("\n--- Saving Results ---")
    
    flat_results = []
    for ds, res in cat_results.items():
        for algo, metrics in res.items():
            flat_results.append({"Category": "Categorical", "Dataset": ds, "Algorithm": algo, **metrics})
            
    for ds, res in sem_results.items():
        for algo, metrics in res.items():
            flat_results.append({"Category": "SEM", "Dataset": ds, "Algorithm": algo, **metrics})
            
    df_res = pd.DataFrame(flat_results)
    res_csv = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    df_res.to_csv(res_csv, index=False)
    print(f"Saved benchmark CSV to {res_csv}")
    
    # 4. Generate plots
    plot_results(cat_results, "Categorical Algorithms Benchmark Time", os.path.join(OUTPUT_DIR, "categorical_benchmark.png"))
    plot_results(sem_results, "SEM Algorithms Benchmark Time", os.path.join(OUTPUT_DIR, "sem_benchmark.png"))
    print("Saved benchmark plots to tests/benchmark_results/")
    print("\nBenchmark completed!")
