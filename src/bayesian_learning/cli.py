import argparse
import time
import os
import pickle
import pandas as pd
from bayesian_learning.categorical_model.learn_categorical import (
    learn as learn_categorical,
    StructureAlgorithm as CatStructureAlgo,
    ParameterAlgorithm as CatParamAlgo
)
from bayesian_learning.SEM_model.learn_sem import (
    learn_structure as learn_sem_structure,
    SEMAlgorithm as SEMAlgo
)
from bayesian_learning.SEM_model.result import GranDAGResult
from bayesian_learning.utils.export import export_categorical_results, export_sem_results
import pyagrum as gum
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Bayesian Learning CLI")
    parser.add_argument("csv_path", type=str, help="Path to the dataset CSV file")

    # Paradigm selection
    parser.add_argument(
        "--type",
        type=str,
        choices=["categorical", "sem"],
        default="categorical",
        help="Type of learning paradigm to use"
    )

    # Structure algorithm
    parser.add_argument(
        "--structure-algo",
        type=str,
        default="hill_climbing",
        help="Structural learning algorithm (hill_climbing, genetic_k2, pc, fci, rfci, structural_em, dagma, dag_gnn, gran_dag)"
    )

    # Parameter algorithm (categorical only)
    parser.add_argument(
        "--parameter-algo",
        type=str,
        default="mle",
        help="Parameter learning algorithm for categorical (mle, em, bayesian_dirichlet_priors, shrinkage_estimator)"
    )

    # Output file
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path prefix to save the output files (e.g., passing 'my_net' will create 'my_net.bif', 'my_net_graph.png', 'my_net_cpts.txt')"
    )

    # Predict Distribution (GraN-DAG only)
    parser.add_argument(
        "--predict-dist",
        action="store_true",
        help="Predict conditional distributions on the dataset (Supported ONLY for gran_dag in SEM mode)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File not found -> {args.csv_path}")
        return

    print(f"Dataset: {args.csv_path}")
    print(f"Paradigm: {args.type.upper()}")
    if args.output:
        print(f"Output Prefix: {args.output}")

    start_time = time.perf_counter()

    if args.type == "categorical":
        if args.predict_dist:
            print("Warning: --predict-dist is only supported for 'gran_dag' in SEM mode. It will be ignored.")

        try:
            struct_algo = CatStructureAlgo[args.structure_algo.upper()]
        except KeyError:
            print(f"Error: Invalid structure algorithm '{args.structure_algo}' for categorical type.")
            print(f"Available: {[e.name for e in CatStructureAlgo]}")
            return

        try:
            param_algo = CatParamAlgo[args.parameter_algo.upper()]
        except KeyError:
            print(f"Error: Invalid parameter algorithm '{args.parameter_algo}'.")
            print(f"Available: {[e.name for e in CatParamAlgo]}")
            return

        print(f"Algorithms: Structure={struct_algo.name}, Parameters={param_algo.name}")
        print("Learning... (please wait)")

        try:
            bn = learn_categorical(args.csv_path, struct_algo, param_algo)
            elapsed = time.perf_counter() - start_time
            print(f"\n--- SUCCESS ---")
            print(f"Time Taken: {elapsed:.3f}s")
            if isinstance(bn, gum.BayesNet):
                print(f"Nodes: {bn.size()}")
                print(f"Arcs (Edges): {bn.sizeArcs()}")
                if args.output:
                    paths = export_categorical_results(bn, args.output)
                    print(f"Saved Network to: {paths[0]}")
                    print(f"Saved Graph Plot to: {paths[1]}")
                    print(f"Saved CPTs to: {paths[2]}")
            else:
                # Fallback if something else is returned
                print(f"Result object type: {type(bn)}")

        except Exception as e:
            print(f"\n--- ERROR ---")
            import traceback
            traceback.print_exc()

    elif args.type == "sem":
        try:
            # Map command line args to the actual enum names
            algo_map = {
                "dagma": "DAGMA",
                "dag_gnn": "DAG_GNN",
                "gran_dag": "GRAN_DAG",
                "dag-gnn": "DAG_GNN",
                "gran-dag": "GRAN_DAG"
            }
            algo_name = algo_map.get(args.structure_algo.lower(), args.structure_algo.upper())
            sem_algo = SEMAlgo[str(algo_name)]
        except KeyError:
            print(f"Error: Invalid structure algorithm '{args.structure_algo}' for SEM type.")
            print(f"Available: {[e.name for e in SEMAlgo]}")
            return

        if args.predict_dist and sem_algo != SEMAlgo.GRAN_DAG:
            print("Error: --predict-dist is ONLY supported for gran_dag algorithm.")
            return

        print(f"Algorithms: Structure={sem_algo.name}")
        print("Learning... (please wait, SEM models can take a long time)")

        try:
            result = learn_sem_structure(args.csv_path, sem_algo)
            elapsed = time.perf_counter() - start_time
            print(f"\n--- SUCCESS ---")
            print(f"Time Taken: {elapsed:.3f}s")
            adj = result.adjacency_matrix
            nodes = adj.shape[0]
            edges = int(np.sum(adj != 0))
            print(f"Nodes: {nodes}")
            print(f"Arcs (Edges): {edges}")

            if args.output:
                paths = export_sem_results(result, args.output)
                print(f"Saved Adjacency Matrix to: {paths[0]}")
                print(f"Saved Graph Plot to: {paths[1]}")

            if args.predict_dist and isinstance(result, GranDAGResult):
                print("\nPredicting distributions with GraN-DAG...")
                df = pd.read_csv(args.csv_path)
                # Predict on the entire dataset
                distributions = result.predict_distribution(df.values)
                print(f"Predicted distributions for {len(distributions)} variables successfully!")
                if args.output:
                    # Optional: save the distributions using pickle if requested
                    dist_out = args.output + "_distributions.pkl"
                    with open(dist_out, "wb") as f:
                        pickle.dump(distributions, f)
                    print(f"Saved PyTorch distributions to: {dist_out}")

        except Exception as e:
            print(f"\n--- ERROR ---")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
