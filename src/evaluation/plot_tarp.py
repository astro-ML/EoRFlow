import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
from cnn import CNN3D_film as CNN 
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global_full as PowerSpectrumDataset
from data_loader import SortedPowerSpectrumDataset
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable, Tuple
import seaborn as sns
from getdist import plots, MCSamples
from scipy.stats import binom  
from tarp import get_tarp_coverage  # Import TARP's coverage function

class InferenceModel:
    """
    A class for inferring the xH values using a CNN + flow model, with 2DPS as input.
    """

    def __init__(self, params: dict, cnn_model: nn.Module, flow_model: nn.Module, data_dirs: list, device: str = 'cpu') -> None:
        self.params = params['plot']
        self.cnn_model = cnn_model.to(device)
        self.flow_model = flow_model.flow.to(device)
        self.device = device

        # Initialize dataset and data loader
        n_labels = params['n_labels']  # Number of xH values
        unsorted_data = PowerSpectrumDataset(data_dirs, exclude_unfinished_reionization=False, exclude_early_reionization=False)
        self.dataset = SortedPowerSpectrumDataset(unsorted_data)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Set up output directory
        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for storing CNN predictions and labels
        self.cnn_pred = None
        self.label = None
        self.n_labels = n_labels
        self.redshifts = params['redshifts']
        self.cond_dims = params['cond_dims']

    def find_cnn_output(self, save_name: str = 'cnn_output_xH.npz') -> Tuple[np.ndarray, np.ndarray]:
        """Run CNN on test data to get predictions or load from file if available."""
        try:
            data = np.load(os.path.join(self.output_dir, save_name))
            self.cnn_pred = data['cnn_pred']
            self.label = data['label']
            logging.info("Loaded CNN output from file.")
        except FileNotFoundError:
            logging.info("Running CNN on test data.")
            num_samples = len(self.dataset)
            self.cnn_pred = np.zeros((num_samples, self.cond_dims))
            self.label = np.zeros((num_samples, self.n_labels))

            # Redshift values (adjust based on your data)
        
            redshifts = torch.tensor(self.redshifts / 10, dtype=torch.float32).to(self.device)

            for i, (ps_data, true_label) in enumerate(self.data_loader):
                ps_data, true_label = ps_data.to(self.device), true_label.to(self.device)

                ps_data = ps_data.unsqueeze(1)

                redshift_batch = redshifts.repeat(ps_data.size(0), 1)  # shape: [batch_size, redshift_dim]
            
                # Forward through CNN
                cnn_output = self.cnn_model(ps_data, redshift_batch)
                
                # Concatenate CNN output and redshift information
                pred = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, cond_dims]

                self.cnn_pred[i] = pred.detach().cpu().numpy()
                self.label[i] = true_label.detach().cpu().numpy()

            np.savez(os.path.join(self.output_dir, save_name), cnn_pred=self.cnn_pred, label=self.label)

        return self.cnn_pred, self.label

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """Calculate test statistics for inferred xH values, including coverage using TARP."""
        self.cnn_pred, self.label = self.find_cnn_output()
        logging.info('Calculating statistics for xH inference.')

        num_sims, num_params = self.label.shape

        # Initialize arrays to store all posterior samples
        samples_all = np.zeros((num_sims, sample_size, num_params), dtype=np.float32)
        
        # Store true parameters
        theta_true = self.label  # shape: (n_sims, n_dims)

        # Debugging: Print shapes before transposition
        print(f"Before transposition:")
        print(f"samples_all shape: {samples_all.shape}")  # Expected: (n_sims, n_samples, n_dims)
        print(f"theta_true shape: {theta_true.shape}")    # Expected: (n_sims, n_dims)

        for i in range(num_sims):
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            condition = torch.Tensor(self.cnn_pred[i]).unsqueeze(0).repeat(sample_size, 1).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            samples = samples.detach().cpu().numpy()
            samples_all[i] = samples

            if (i + 1) % 100 == 0 or (i + 1) == num_sims:
                logging.info(f"Processed {i + 1}/{num_sims} samples.")

        # Transpose samples_all to (n_samples, n_sims, n_dims)
        samples_all_transposed = np.transpose(samples_all, (1, 0, 2))  # (n_samples, n_sims, n_dims)

        # Debugging: Print shapes after transposition
        print(f"After transposition:")
        print(f"samples_all_transposed shape: {samples_all_transposed.shape}")  # Expected: (n_samples, n_sims, n_dims)
        print(f"theta_true shape: {theta_true.shape}")                        # Expected: (n_sims, n_dims)

        # Save all samples and true parameters for later use
        np.savez(os.path.join(self.output_dir, 'all_posterior_samples.npz'),
                samples_all=samples_all_transposed,
                theta_true=theta_true)
        logging.info('All posterior samples and true parameters saved.')

        # Use TARP to compute coverage
        try:
            ecp, alpha = get_tarp_coverage(
                samples=samples_all_transposed,  # shape: (n_samples, n_sims, n_dims)
                theta=theta_true,                # shape: (n_sims, n_dims)
                references='random',             # or 'rank', 'hpd' as per TARP documentation
                metric='euclidean',              # distance metric, adjust if needed
                norm=False,                       # whether to normalize the data
                seed=5                            # for reproducibility
            )
        except ValueError as ve:
            print(f"Error in get_tarp_coverage: {ve}")
            logging.error(f"Error in get_tarp_coverage: {ve}")
            return

        # Debugging: Print coverage shapes
        print(f"ecp shape: {ecp.shape}")    # Expected: same as alpha
        print(f"alpha shape: {alpha.shape}")# Expected: same as ecp

        # Save coverage results
        np.savez(os.path.join(self.output_dir, 'tarp_coverage.npz'),
                ecp=ecp,
                alpha=alpha)
        logging.info('TARP coverage results saved.')

    def plot_tarp(self, coverage_file: str = 'tarp_coverage.npz', output_name: str = 'coverage_tarp_plot.pdf') -> None:
        """Plot coverage using TARP results."""
        logging.info('Plotting coverage using TARP results.')
        
        # Load coverage results
        data = np.load(os.path.join(self.output_dir, coverage_file))
        ecp = data['ecp']      # Expected Coverage Points
        alpha = data['alpha']  # Corresponding Credibility Levels

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Ideal Coverage')
        plt.plot(alpha, ecp, marker='o', linestyle='-', color='blue', label='TARP Coverage')
        plt.xlabel('Credibility Level (%)', fontsize=14)
        plt.ylabel('Expected Coverage (%)', fontsize=14)
        plt.title('Coverage Plot Using TARP', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()
        logging.info("Coverage plot saved using TARP.")

    def main(self) -> None:
        """Main function to run inference and create plots."""
        self.calc_statistics(output_name='inference_statistics_xH.npz', sample_size=1000)
        
        # Plot coverage using TARP
        self.plot_tarp(coverage_file='tarp_coverage.npz', output_name='coverage_tarp_plot.pdf')
        
        logging.info("Inference and plotting complete.")

# Define the model directory and paths
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pure_TomsData_pretrained_noFilter'

redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
        7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
       10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
       12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
       15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
       17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])

# Parameters dictionary setup
params = {
    'n_labels': 30,         # Number of xH values
    'redshifts': redshift_values,     # Number of redshift values
    'cond_dims': 30+30,        # CNN output size + redshift_dim
    'plot': {
        'plot_dir': os.path.join(model_dir, 'plots_tarp'),
    }
}

# Load the pretrained CNN and flow models
cnn_model = CNN()  # Use your modified CNN model
cnn_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_cnn_model.pth')))
cnn_model.eval()

model_params = {
    'flow': {
        'n_dim': params['n_labels'],
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': params['cond_dims'],
        'dropout': 0.0, 
        'load': False,
        'model_location': 'trained_model.pth',
    }
}

flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.load_state_dict(torch.load(os.path.join(model_dir, 'best_flow_model.pth')))
flow_model.flow.eval()

# Create inference model instance
inference_model = InferenceModel(
    params=params,
    cnn_model=cnn_model,
    flow_model=flow_model,
    data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10'],
    #data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise',
    #           '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'],
    #data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise',
    #           '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro',
    #           '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Run inference
inference_model.main()