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
    
    # 1. Initialize the Generator with the BayesNet
    generator = gum.BNDatabaseGenerator(bn)
    
    # 2. (Optional) Set the column order. 
    # By default, it might be random or topological.
    generator.setTopologicalVarOrder()
    
    print(f"Drawing {n_samples} samples...")
    # 3. Draw the samples. 
    # Note: If the BN has very rare events (like our Tsunami BN), 
    # it generates them according to their true distributions.
    generator.drawSamples(n_samples)
    
    # 4. Export to CSV
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
