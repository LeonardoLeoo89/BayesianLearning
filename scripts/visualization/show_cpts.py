import pyagrum as gum
import os

def export_cpts_to_txt(bn: gum.BayesNet, output_file: str):
    with open(output_file, 'w') as f:
        f.write(f"=== CPTs for {bn.property('name') if 'name' in bn.properties() else 'Network'} ===\n\n")
        
        for node in bn.names():
            f.write(f"Node: {node}\n")
            f.write("-" * (len(node) + 6) + "\n")
            f.write(str(bn.cpt(node)))
            f.write("\n\n")
            
if __name__ == "__main__":
    os.makedirs("tests/cpt_tables", exist_ok=True)
    
    # Load networks
    tsunami_bn = gum.loadBN("generated_bns/tsunami.bif")
    allergy_bn = gum.loadBN("generated_bns/allergy.bif")
    train_bn = gum.loadBN("generated_bns/train_delay.bif")
    
    export_cpts_to_txt(tsunami_bn, "tests/cpt_tables/tsunami_cpts.txt")
    export_cpts_to_txt(allergy_bn, "tests/cpt_tables/allergy_cpts.txt")
    export_cpts_to_txt(train_bn, "tests/cpt_tables/train_delay_cpts.txt")
    
    print("Exported all CPTs to tests/cpt_tables/")
