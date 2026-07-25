import numpy as np
import networkx as nx

class SEMResult:
    """Universal data structure for learned DAGs from SEM algorithms."""
    
    def __init__(self, adjacency_matrix: np.ndarray, node_names: list[str] | None = None):
        """
        Initializes a generic SEM result.
        
        Args:
            adjacency_matrix: A 2D numpy array of shape (d, d) where a non-zero value
                              at (i, j) indicates a directed edge from i to j.
            node_names: Optional list of variable names. Defaults to ['X0', 'X1', ...].
        """
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        
        if node_names is not None:
            if len(node_names) != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} node names, got {len(node_names)}")
            self.node_names = node_names
        else:
            self.node_names = [f"X{i}" for i in range(self.num_nodes)]
            
    def to_networkx(self) -> nx.DiGraph:
        """
        Converts the adjacency matrix to a universal NetworkX DiGraph.
        This provides a standardized format for plotting and structural analysis.
        """
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        nx.relabel_nodes(G, {i: name for i, name in enumerate(self.node_names)}, copy=False)
        return G

class GranDAGResult(SEMResult):
    """Specific result class for GraN-DAG that extends the universal DAG structure."""
    
    def __init__(self, adjacency_matrix: np.ndarray, trained_model, node_names: list[str] | None = None):
        """
        Args:
            adjacency_matrix: The learned weighted DAG structure.
            trained_model: The PyTorch neural network model trained by GraN-DAG.
            node_names: Optional list of variable names.
        """
        super().__init__(adjacency_matrix, node_names)
        self.trained_model = trained_model
        
    def predict_distribution(self, observations: np.ndarray):
        """
        Query probabilities/distributions using the trained neural network.
        
        Args:
            observations: Input data/samples.
            
        Returns:
            The predicted conditional distributions.
        """
        import torch
        
        # Move to the correct device and cast to correct dtype
        device = next(self.trained_model.parameters()).device
        dtype = next(self.trained_model.parameters()).dtype
        
        # GraN-DAG expects 2D tensors
        if not isinstance(observations, torch.Tensor):
            x = torch.tensor(observations, dtype=dtype, device=device)
        else:
            x = observations.to(device=device, dtype=dtype)
            
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        # 1. Retrieve the learned parameters (weights, biases, and any extra parameters like variance)
        weights, biases, extra_params = self.trained_model.get_parameters(mode="wbx")
        
        # 2. Forward pass to get density parameters (e.g., predicted means)
        density_params = self.trained_model.forward_given_params(x, weights, biases)
        
        # 3. Apply any necessary transformations to extra parameters (e.g., exponentiating log_std)
        if len(extra_params) != 0:
            transformed_extra = self.trained_model.transform_extra_params(self.trained_model.extra_params)
        else:
            transformed_extra = []
            
        distributions = []
        
        # 4. Construct the PyTorch probability distribution for each variable
        for i in range(self.trained_model.num_vars):
            # Extract the specific density parameters for this variable
            density_param = list(torch.unbind(density_params[i], 1))
            
            # Append any global extra parameters for this variable (like std dev)
            if len(transformed_extra) != 0:
                density_param.extend(list(torch.unbind(transformed_extra[i], 0)))
                
            # Create the PyTorch Distribution (e.g. Normal(mean, std))
            conditional_dist = self.trained_model.get_distribution(density_param)
            distributions.append(conditional_dist)
            
        return distributions
