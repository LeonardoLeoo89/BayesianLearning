import os
import argparse
import tempfile
import numpy as np
import pandas as pd
from ..result import SEMResult, GranDAGResult
from .base import SEMWrapper

class GraNDAGWrapper(SEMWrapper):
    """Wrapper for the GraN-DAG model running natively within the process."""
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Hyperparameters to pass to GraN-DAG (e.g. num_train_iter=100000, lr=1e-3, etc.)
                      Defaults are set internally if not provided.
        """
        self.kwargs = kwargs
        
    def learn(self, data: pd.DataFrame) -> SEMResult:
        """Learns the DAG structure using GraN-DAG."""
        
        # We import here so we don't fail immediately if GraN-DAG is not installed
        from gran_dag.main import train_from_array
        
        n_samples, n_vars = data.shape
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = os.path.join(temp_dir, "exp")
            
            # Default options expected by GraN-DAG
            opt = argparse.Namespace(
                data_path=temp_dir,
                exp_path=exp_dir,
                i_dataset=1,
                num_vars=n_vars,
                train=True,
                to_dag=True,
                model="NonLinGauss",
                num_layers=2,
                hid_dim=10,
                nonlin="leaky-relu",
                norm_prod="none",
                square_prod=False,
                pns=False,
                pns_thresh=0.75,
                num_neighbors=None,
                cam_pruning=False,
                retrain=False,
                random_seed=42,
                lr=1e-3,
                lr_reinit=None,
                gpu=False,
                float=False,
                train_samples=0.8,
                test_samples=None,
                normalize_data=False,
                num_train_iter=100000,
                train_batch_size=64,
                mu_init=0.001,
                lambda_init=0.0,
                optimizer="rmsprop",
                edge_clamp_range=0.0001,
                no_w_adjs_log=True,
                stop_crit_win=100,
                omega_lambda=0.0001,
                omega_mu=0.9,
                h_threshold=1e-8,
                plot_freq=1000000,
                jac_thresh=True
            )
            
            # Apply user-provided kwargs
            for key, value in self.kwargs.items():
                setattr(opt, key, value)
                
            # Run natively without subprocess
            model = train_from_array(opt, data.values, adjacency_array=np.zeros((n_vars, n_vars)))
            
            # Get the output graph
            pred_dag_path = os.path.join(exp_dir, "to-dag", "DAG.npy")
            if os.path.exists(pred_dag_path):
                W_est = np.load(pred_dag_path)
            else:
                W_est = model.adjacency.detach().cpu().numpy()
            
            if W_est.shape != (n_vars, n_vars):
                W_est = W_est.reshape((n_vars, n_vars))
                
            return GranDAGResult(W_est, model, node_names=list(data.columns))
