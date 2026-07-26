import pyagrum as gum
import os

def create_tsunami_network() -> gum.BayesNet:
    """Creates a Low Complexity BN for Tsunami Risk"""
    bn = gum.BayesNet("TsunamiRisk")
    
    # Add nodes with their domains
    bn.add(gum.LabelizedVariable("SubmarineEarthquake", "Submarine Earthquake", 2))
    bn.changeVariableLabel("SubmarineEarthquake", "0", "False")
    bn.changeVariableLabel("SubmarineEarthquake", "1", "True")
    
    bn.add(gum.LabelizedVariable("CoastalLandslide", "Coastal Landslide", 2))
    bn.changeVariableLabel("CoastalLandslide", "0", "False")
    bn.changeVariableLabel("CoastalLandslide", "1", "True")
    
    bn.add(gum.LabelizedVariable("Tsunami", "Tsunami Generated", 2))
    bn.changeVariableLabel("Tsunami", "0", "False")
    bn.changeVariableLabel("Tsunami", "1", "True")
    
    bn.add(gum.LabelizedVariable("TsunamiWarning", "Tsunami Warning System", 2))
    bn.changeVariableLabel("TsunamiWarning", "0", "Inactive")
    bn.changeVariableLabel("TsunamiWarning", "1", "Active")
    
    # Add edges
    bn.addArc("SubmarineEarthquake", "Tsunami")
    bn.addArc("CoastalLandslide", "Tsunami")
    bn.addArc("Tsunami", "TsunamiWarning")
    bn.addArc("SubmarineEarthquake", "TsunamiWarning")
    
    # Populate CPTs
    bn.cpt("SubmarineEarthquake").fillWith([0.99, 0.01])
    bn.cpt("CoastalLandslide").fillWith([0.98, 0.02])
    
    bn.cpt("Tsunami")[{'SubmarineEarthquake': 'False', 'CoastalLandslide': 'False'}] = [0.999, 0.001]
    bn.cpt("Tsunami")[{'SubmarineEarthquake': 'True', 'CoastalLandslide': 'False'}] = [0.20, 0.80]
    bn.cpt("Tsunami")[{'SubmarineEarthquake': 'False', 'CoastalLandslide': 'True'}] = [0.30, 0.70]
    bn.cpt("Tsunami")[{'SubmarineEarthquake': 'True', 'CoastalLandslide': 'True'}] = [0.05, 0.95]
    
    bn.cpt("TsunamiWarning")[{'Tsunami': 'False', 'SubmarineEarthquake': 'False'}] = [0.99, 0.01]
    bn.cpt("TsunamiWarning")[{'Tsunami': 'True', 'SubmarineEarthquake': 'False'}] = [0.05, 0.95]
    bn.cpt("TsunamiWarning")[{'Tsunami': 'False', 'SubmarineEarthquake': 'True'}] = [0.10, 0.90] # Early seismometer trigger
    bn.cpt("TsunamiWarning")[{'Tsunami': 'True', 'SubmarineEarthquake': 'True'}] = [0.01, 0.99]
    
    return bn

def create_allergy_network() -> gum.BayesNet:
    """Creates a Medium Complexity BN for Cross-reactive Allergies"""
    bn = gum.BayesNet("AllergyCrossReactivity")
    
    nodes = ["Atopy", "DustMiteAllergy", "PollenAllergy", "BirchPollenAllergy", 
             "AppleAllergy", "HazelnutAllergy", "Asthma", "AllergicRhinitis"]
             
    for node in nodes:
        bn.add(gum.LabelizedVariable(node, node, 2))
        bn.changeVariableLabel(node, "0", "No")
        bn.changeVariableLabel(node, "1", "Yes")
        
    edges = [
        ("Atopy", "DustMiteAllergy"),
        ("Atopy", "PollenAllergy"),
        ("PollenAllergy", "BirchPollenAllergy"),
        ("PollenAllergy", "AllergicRhinitis"),
        ("BirchPollenAllergy", "AppleAllergy"),
        ("BirchPollenAllergy", "HazelnutAllergy"),
        ("DustMiteAllergy", "Asthma")
    ]
    
    for u, v in edges:
        bn.addArc(u, v)
        
    # Populate CPTs
    bn.cpt("Atopy").fillWith([0.75, 0.25])
    
    bn.cpt("DustMiteAllergy")[{'Atopy': 'No'}] = [0.90, 0.10]
    bn.cpt("DustMiteAllergy")[{'Atopy': 'Yes'}] = [0.40, 0.60]
    
    bn.cpt("PollenAllergy")[{'Atopy': 'No'}] = [0.85, 0.15]
    bn.cpt("PollenAllergy")[{'Atopy': 'Yes'}] = [0.30, 0.70]
    
    bn.cpt("BirchPollenAllergy")[{'PollenAllergy': 'No'}] = [0.98, 0.02]
    bn.cpt("BirchPollenAllergy")[{'PollenAllergy': 'Yes'}] = [0.60, 0.40]
    
    bn.cpt("AllergicRhinitis")[{'PollenAllergy': 'No'}] = [0.95, 0.05]
    bn.cpt("AllergicRhinitis")[{'PollenAllergy': 'Yes'}] = [0.20, 0.80]
    
    bn.cpt("AppleAllergy")[{'BirchPollenAllergy': 'No'}] = [0.98, 0.02]
    bn.cpt("AppleAllergy")[{'BirchPollenAllergy': 'Yes'}] = [0.50, 0.50] # Cross-reactivity
    
    bn.cpt("HazelnutAllergy")[{'BirchPollenAllergy': 'No'}] = [0.99, 0.01]
    bn.cpt("HazelnutAllergy")[{'BirchPollenAllergy': 'Yes'}] = [0.70, 0.30] # Cross-reactivity
    
    bn.cpt("Asthma")[{'DustMiteAllergy': 'No'}] = [0.95, 0.05]
    bn.cpt("Asthma")[{'DustMiteAllergy': 'Yes'}] = [0.50, 0.50]
        
    return bn

def create_train_delay_network() -> gum.BayesNet:
    """Creates a High Complexity BN for Train Delay Predictions"""
    bn = gum.BayesNet("TrainDelay")
    
    # Define variables and domains
    vars_def = {
        "Season": ["Winter", "Spring", "Summer", "Autumn"],
        "Weather": ["Clear", "Rain", "Snow", "Storm"],
        "TimeOfDay": ["MorningRush", "Midday", "EveningRush", "Night"],
        "PassengerVolume": ["Low", "Normal", "High"],
        "TrackIncident": ["False", "True"],
        "InfraFailure": ["False", "True"],
        "SpeedRestriction": ["False", "True"],
        "HubCongestion": ["False", "True"],
        "DepartureDelay": ["None", "Minor", "Major"],
        "ArrivalDelay": ["None", "Minor", "Major"],
        "CompensationClaim": ["False", "True"]
    }
    
    for name, domain in vars_def.items():
        bn.add(gum.LabelizedVariable(name, name, len(domain)))
        for i, val in enumerate(domain):
            bn.changeVariableLabel(name, str(i), val)
            
    # Add edges
    edges = [
        ("Season", "Weather"),
        ("TimeOfDay", "PassengerVolume"),
        ("Weather", "TrackIncident"),
        ("Weather", "InfraFailure"),
        ("Weather", "SpeedRestriction"),
        ("TrackIncident", "SpeedRestriction"),
        ("TrackIncident", "InfraFailure"),
        ("InfraFailure", "DepartureDelay"),
        ("PassengerVolume", "HubCongestion"),
        ("HubCongestion", "DepartureDelay"),
        ("DepartureDelay", "ArrivalDelay"),
        ("SpeedRestriction", "ArrivalDelay"),
        ("ArrivalDelay", "CompensationClaim")
    ]
    
    for u, v in edges:
        bn.addArc(u, v)
        
    # Populate CPTs automatically since this network has a huge state space
    bn.generateCPTs()
        
    return bn

if __name__ == "__main__":
    tsunami_bn = create_tsunami_network()
    allergy_bn = create_allergy_network()
    train_bn = create_train_delay_network()
    
    print(f"Generated Tsunami Risk BN with {tsunami_bn.size()} nodes and {tsunami_bn.sizeArcs()} arcs.")
    print(f"Generated Allergy Cross-Reactivity BN with {allergy_bn.size()} nodes and {allergy_bn.sizeArcs()} arcs.")
    print(f"Generated Train Delay BN with {train_bn.size()} nodes and {train_bn.sizeArcs()} arcs.")
    
    # Save BIF files (Bayesian Interchange Format)
    os.makedirs("generated_bns", exist_ok=True)
    gum.saveBN(tsunami_bn, "generated_bns/tsunami.bif")
    gum.saveBN(allergy_bn, "generated_bns/allergy.bif")
    gum.saveBN(train_bn, "generated_bns/train_delay.bif")
    print("Saved BIF files (including generated CPTs) to 'generated_bns' directory.")
