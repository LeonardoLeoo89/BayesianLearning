import os
import pandas as pd
import pyagrum as gum
from bayesian_learning.categorical_model.learn_categorical import learn_parameters, ParameterAlgorithm

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

def main():
    datasets = ["tsunami", "allergy", "train_delay"]
    algos = [
        ("MLE", ParameterAlgorithm.MLE),
        ("BDeu", ParameterAlgorithm.BAYESIAN_DIRICHLET_PRIORS),
        ("Shrinkage", ParameterAlgorithm.SHRINKAGE_ESTIMATOR)
    ]

    results = []

    for ds_name in datasets:
        dataset_path = f"data/categorical/{ds_name}_samples_subset_500.csv"
        bif_path = f"data/ground_truth/{ds_name}.bif"
        if not os.path.exists(dataset_path) or not os.path.exists(bif_path):
            continue

        true_bn = gum.loadBN(bif_path)
        sorted_true_bn = sort_bn(true_bn)

        df = pd.read_csv(dataset_path)
        nodes = list(df.columns)

        # Create aligned BN structure
        aligned_bn = gum.BayesNet()
        for n in nodes:
            aligned_bn.add(sorted_true_bn.variableFromName(n))
        for u, v in sorted_true_bn.arcs():
            aligned_bn.addArc(sorted_true_bn.variable(u).name(), sorted_true_bn.variable(v).name())

        for algo_name, algo_enum in algos:
            try:
                pred_bn_fitted = learn_parameters(dataset_path, aligned_bn.dag(), algo_enum)

                sorted_pred_bn = sort_bn(pred_bn_fitted)

                dist = gum.ExactBNdistance(sorted_true_bn, sorted_pred_bn)
                res = dist.compute()

                results.append({
                    "Dataset": ds_name,
                    "Algorithm": algo_name,
                    "KL_Div": res.get('klPQ', float('nan')),
                    "Hellinger": res.get('hellinger', float('nan')),
                    "Bhattacharyya": res.get('bhattacharya', float('nan')),
                    "Jensen_Shannon": res.get('jensen-shannon', float('nan'))
                })
            except Exception as e:
                print(f"Failed {ds_name} {algo_name}: {e}")

    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
