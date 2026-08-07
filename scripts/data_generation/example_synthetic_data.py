import pyagrum as gum
import os

def generate_data(bif_file: str, output_csv: str, n_samples: int = 1000):
    """
    Generates synthetic categorical data from a Bayesian Network.

    Args:
        bif_file: Path to the .bif network file
        output_csv: Path to save the generated CSV
        n_samples: Number of samples to draw
    """
    print(f"Loading network from {bif_file}...")
    bn = gum.loadBN(bif_file)

    generator = gum.BNDatabaseGenerator(bn)

    generator.setTopologicalVarOrder()

    print(f"Drawing {n_samples} samples...")
    generator.drawSamples(n_samples)

    generator.toCSV(output_csv)
    print(f"Successfully generated {n_samples} samples and saved to {output_csv}\n")

if __name__ == "__main__":
    os.makedirs("data/categorical", exist_ok=True)

    generate_data(
        bif_file="data/ground_truth/tsunami.bif",
        output_csv="data/categorical/tsunami_samples.csv",
        n_samples=5000
    )

    generate_data(
        bif_file="data/ground_truth/allergy.bif",
        output_csv="data/categorical/allergy_samples.csv",
        n_samples=5000
    )

    generate_data(
        bif_file="data/ground_truth/train_delay.bif",
        output_csv="data/categorical/train_delay_samples.csv",
        n_samples=5000
    )

    print("All synthetic data generated! You can load these CSVs into pandas or your BN learners.")
