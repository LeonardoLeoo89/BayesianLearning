import pyagrum as gum
import matplotlib.pyplot as plt
import os
import math

def plot_marginals(bn: gum.BayesNet, title: str, output_path: str):
    ie = gum.LazyPropagation(bn)
    ie.makeInference()

    nodes = bn.names()
    n_nodes = len(nodes)

    cols = 3
    rows = math.ceil(n_nodes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle(f'Marginal Probabilities: {title}', fontsize=16, y=1.02)

    axes = axes.flatten()

    for i, node in enumerate(nodes):
        ax = axes[i]

        posterior = ie.posterior(node)

        labels = [bn.variable(node).label(idx) for idx in range(bn.variable(node).domainSize())]
        probs = [posterior[idx] for idx in range(bn.variable(node).domainSize())]

        bars = ax.bar(labels, probs, color='skyblue', edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

        ax.set_title(node, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved marginals plot to {output_path}")

if __name__ == "__main__":
    os.makedirs("results/probability_plots", exist_ok=True)

    tsunami_bn = gum.loadBN("data/ground_truth/tsunami.bif")
    allergy_bn = gum.loadBN("data/ground_truth/allergy.bif")
    train_bn = gum.loadBN("data/ground_truth/train_delay.bif")

    print("Generating marginal probability plots...")
    plot_marginals(tsunami_bn, "Tsunami Risk Network", "results/probability_plots/tsunami_marginals.png")
    plot_marginals(allergy_bn, "Allergy Cross-Reactivity Network", "results/probability_plots/allergy_marginals.png")
    plot_marginals(train_bn, "Train Delay Prediction Network", "results/probability_plots/train_delay_marginals.png")

    print("\nSuccessfully generated probability plots!")
