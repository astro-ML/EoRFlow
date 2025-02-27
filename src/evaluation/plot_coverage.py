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
from cnn import CNN3D_15 as CNN 
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global as PowerSpectrumDataset
#from data_loader import PowerSpectrumDataset_global_10param as PowerSpectrumDataset
from data_loader import SortedPowerSpectrumDataset
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable, Tuple
import seaborn as sns
from getdist import plots, MCSamples
from scipy.stats import binom  
from tarp import get_tarp_coverage  # Ensure this module is accessible

# Configure logging to display information
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class InferenceModel:
    """
    A class for inferring the xH values using a CNN + flow model, with 2DPS as input.
    Includes both manual and rank-based coverage calculations.
    """

    def __init__(self, params: dict, cnn_model: nn.Module, flow_model: nn.Module, data_dirs: list, device: str = 'cpu') -> None:
        self.params = params['plot']
        self.cnn_model = cnn_model.to(device)
        self.flow_model = flow_model  
        self.model = self.flow_model.flow.to(device)
        self.device = device 

        # Initialize dataset and data loader
        n_labels = params['n_labels']  # Number of xH values
        unsorted_data = PowerSpectrumDataset(data_dirs, exclude_unfinished_reionization=False, exclude_early_reionization=False)
        self.dataset = SortedPowerSpectrumDataset(unsorted_data)
        #self.dataset = unsorted_data
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
            #redshifts = torch.tensor(self.redshifts / 10, dtype=torch.float32).to(self.device)

            for i, (ps_data, true_label, redshift_batch) in enumerate(self.data_loader):
                ps_data, true_label, redshift_batch= ps_data.to(self.device), true_label.to(self.device), redshift_batch.to(self.device)
                #ps_data, true_label= ps_data.to(self.device), true_label.to(self.device)

                ps_data = ps_data.unsqueeze(1)

                #redshift_batch = redshifts.repeat(ps_data.size(0), 1)  # shape: [batch_size, cond_dims]
            
                # Forward through CNN
                cnn_output = self.cnn_model(ps_data, redshift_batch)
                
                # Concatenate CNN output and redshift information
                pred = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, cond_dims]

                self.cnn_pred[i] = pred.detach().cpu().numpy()
                self.label[i] = true_label.detach().cpu().numpy()

            np.savez(os.path.join(self.output_dir, save_name), cnn_pred=self.cnn_pred, label=self.label)
            logging.info("CNN outputs saved to file.")

        return self.cnn_pred, self.label

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """Calculate test statistics for inferred xH values, including manual coverage."""
        self.cnn_pred, self.label = self.find_cnn_output()
        logging.info('Calculating statistics for xH inference.')

        num_samples, num_params = self.label.shape
        mean = np.zeros((num_samples, num_params))
        lower = np.zeros((num_samples, num_params))
        upper = np.zeros((num_samples, num_params))
        rank = np.zeros((num_samples, num_params))

        # Define confidence levels
        confidence_levels = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 
                             50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]  # Percentages

        # Initialize coverage counters
        coverage_counts = {level: np.zeros(num_params) for level in confidence_levels}

        for i in range(num_samples):
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            condition = torch.Tensor(self.cnn_pred[i]).unsqueeze(0).repeat(sample_size, 1).to(self.device)
            samples, _ = self.model(z, c=[condition], rev=True)
            #samples = torch.clamp(samples,0,1)
            samples = samples.detach().cpu().numpy()

            # Compute statistics for each parameter
            mean[i] = np.mean(samples, axis=0)
            lower[i] = np.percentile(samples, 16, axis=0)
            upper[i] = np.percentile(samples, 84, axis=0)

            # Calculate rank statistics
            for j in range(num_params):
                rank[i, j] = np.sum(samples[:, j] < self.label[i, j])

            # Compute coverage for each confidence level
            true_values = self.label[i]
            for level in confidence_levels:
                lower_bound = np.percentile(samples, (100 - level) / 2, axis=0)
                upper_bound = np.percentile(samples, 100 - (100 - level) / 2, axis=0)
                coverage = (true_values >= lower_bound) & (true_values <= upper_bound)
                coverage_counts[level] += coverage.astype(int)

            if (i+1) % 100 == 0 or (i+1) == num_samples:
                logging.info(f"Processed {i+1}/{num_samples} samples.")

        # Calculate empirical coverage percentages
        empirical_coverage = {level: (coverage_counts[level] / num_samples) * 100 for level in confidence_levels}

        # Save statistics including coverage
        np.savez(os.path.join(self.output_dir, output_name),
                mean=mean, lower=lower, upper=upper, label=self.label, rank=rank, coverage=empirical_coverage)
        logging.info('Statistics saved with coverage.')


    def calc_rank_coverage(self, sample_size: int = 1000, output_name: str = 'rank_coverage.npz') -> None:
        """
        Compute rank-based coverage using the log-probabilities of the true labels
        vs. posterior samples.

        Args:
            sample_size: Number of flow-based posterior samples to draw per data example.
            output_name: File name for saving rank coverage arrays.
        """
        if self.cnn_pred is None or self.label is None:
            logging.info("CNN predictions or labels not found, running find_cnn_output()...")
            self.find_cnn_output()

        num_samples, num_params = self.label.shape

        # Arrays to store log-probs
        # shape (num_samples,) for the entire vector's log-prob
        param_logprobs = np.zeros(num_samples)

        # shape (num_samples, sample_size) for posterior samples
        sample_logprobs = np.zeros((num_samples, sample_size))

        # Evaluate everything in no_grad mode
        with torch.no_grad():
            for i in range(num_samples):
                # 1) True label x^true and condition c
                x_true = torch.tensor(self.label[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                c_true = torch.tensor(self.cnn_pred[i], dtype=torch.float32, device=self.device).unsqueeze(0)

                # 2) log-prob of the *true* label
                #    Make sure your flow class has .log_prob(x, c) returning shape [batch_size].
                logprob_true = self.flow_model.log_prob(x_true, c_true)
                # logprob_true is shape [1], so we take item()
                param_logprobs[i] = logprob_true.item()

                # 3) Sample from flow & compute sample log-probs
                z = torch.randn((sample_size, num_params), device=self.device)
                c_sample = c_true.repeat(sample_size, 1)  # replicate same condition
                x_samples, _ = self.model(z, c=[c_sample], rev=True)  # shape [sample_size, num_params]
                #x_samples = torch.clamp(x_samples,0,1)
                # 4) log-prob of the posterior samples
                #    log_prob() expects shape [batch_size, n_dim]
                #    c_sample is [sample_size, cond_dims]
                logprob_samples = self.flow_model.log_prob(x_samples, c_sample)  # shape [sample_size]
                sample_logprobs[i] = logprob_samples.cpu().numpy()

                if (i+1) % 100 == 0 or (i+1) == num_samples:
                    logging.info(f"[Rank Coverage] Processed {i+1}/{num_samples} samples.")

        # 5) Compute ranks: 
        #    For each i, rank[i] = fraction of posterior samples that have smaller log-prob than the true log-prob
        #    or equivalently: rank[i] = mean( logprob_true > logprob_sample ).
        #    The snippet you showed does (true_logprob > sample_logprobs).mean(1).
        #    Adjust sign if you want “higher log-prob is better or lower is better.”

        ranks = (param_logprobs[:, None] > sample_logprobs).mean(axis=1)
        #ranks_per_param = (param_logprobs[:, None, :] > sample_logprobs).mean(axis=1)  # Shape: (n_sims, n_dims)

        # 6) Optionally store or plot the rank distribution
        #    One common approach: we plot the empirical CDF of (1 - ranks) vs. uniform[0,1].
        bins = np.linspace(0, 1, 50)
        """
        plt.figure(figsize=(8, 8))
        for i in range(ranks_per_param.shape[1]):  # Loop over parameters
            quantiles = np.quantile(1 - ranks_per_param[:, i], bins)
            plt.plot(quantiles, bins, label=f'Parameter {i+1}')

        plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Ideal Coverage')
        plt.xlabel('Credibility Level (%)')
        plt.ylabel('Expected Coverage (%)')
        plt.legend()
        plt.title('Log Probability Coverage per Parameter')
        plot_path = os.path.join(self.output_dir, 'rank_coverage_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Rank coverage plot saved to {plot_path}.")



        # quantiles of 1 - ranks at each bin
        """
        quantiles = np.quantile(1 - ranks, bins)

        plt.figure(figsize=(5, 5))
        plt.plot(quantiles, bins, label='Empirical coverage', alpha=0.9)
        plt.plot([0,1], [0,1], 'k--', label='Ideal coverage')
        plt.xlabel('Quantile of (1 - rank)')
        plt.ylabel('Cumulative probability')
        plt.legend()
        plt.title('Rank-based Coverage using Log Probabilities')
        plot_path = os.path.join(self.output_dir, 'rank_coverage_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Rank coverage plot saved to {plot_path}.")
        
    def calc_rank_coverage_per_param(self, sample_size: int = 1000, output_name: str = 'rank_coverage_per_param.npz') -> None:
        """
        Compute rank-based coverage *per parameter* using the per-parameter log-probabilities
        of the true labels vs. posterior samples.
        """
        if self.cnn_pred is None or self.label is None:
            logging.info("CNN predictions or labels not found, running find_cnn_output()...")
            self.find_cnn_output()

        num_sims, num_params = self.label.shape

        # We'll store the ranks in shape [num_sims, num_params]
        ranks_per_param = np.zeros((num_sims, num_params))

        with torch.no_grad():
            for i in range(num_sims):
                # 1) True label x^true and condition c
                x_true = torch.tensor(self.label[i], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, n_params]
                c_true = torch.tensor(self.cnn_pred[i], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, cond_dims]

                # 2) log-prob of the *true* label (PER DIMENSION)
                logprob_true_per_dim = self.flow_model.log_prob_per_dim(x_true, c_true)  # [1, n_params]

                # 3) Sample from flow & compute sample log-probs (PER DIMENSION)
                z = torch.randn((sample_size, num_params), device=self.device)
                c_sample = c_true.repeat(sample_size, 1)  # replicate same condition
                x_samples, _ = self.model(z, c=[c_sample], rev=True)  # [sample_size, n_params]
                #x_samples = torch.clamp(x_samples,0,1)
                logprob_samples_per_dim = self.flow_model.log_prob_per_dim(x_samples, c_sample)  # [sample_size, n_params]

                # 4) For each parameter d, compute rank:
                # ranks_per_param[i, d] = fraction of sample log-probs that are < true log-prob
                # So if we interpret "higher log-prob => better," we do:
                #    rank = mean( logprob_true > logprob_sample )
                # shape: [sample_size, n_params]
                for d in range(num_params):
                    # Compare the single true log-prob for dimension d to all sample log-probs in dimension d
                    ranks_per_param[i, d] = np.mean(
                        logprob_true_per_dim[0, d].item() > logprob_samples_per_dim[:, d].cpu().numpy()
                    )

                if (i+1) % 100 == 0 or (i+1) == num_sims:
                    logging.info(f"[Per-Parameter Rank Coverage] Processed {i+1}/{num_sims} samples.")

        # 5) Now we have ranks_per_param shape [num_sims, n_params].
        # Let's do an empirical coverage plot for each parameter.
        bins = np.linspace(0, 1, 50)
        plt.figure(figsize=(8, 8))
        
        for d in range(num_params):
            # For parameter d, ranks_per_param[:, d] shape: [num_sims]
            # We typically plot the CDF of 1 - ranks
            quantiles = np.quantile(1.0 - ranks_per_param[:, d], bins)
            plt.plot(quantiles, bins, label=f'Param {d+1}')

        plt.plot([0, 1], [0, 1], 'k--', label='Ideal Coverage')
        plt.xlabel('Quantile of (1 - rank)')
        plt.ylabel('Cumulative Probability')
        plt.title('Log-Probability Coverage per Parameter')
        plt.legend()
        plot_path = os.path.join(self.output_dir, 'rank_coverage_per_param.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Per-parameter rank coverage plot saved to {plot_path}.")

        # Optionally, save the ranks for later analysis
        np.savez(os.path.join(self.output_dir, output_name),
                ranks_per_param=ranks_per_param)
        logging.info(f"Saved per-parameter ranks to {output_name}.")
    

    def main(self) -> None:
        """Main function to run inference and create plots."""
        # Calculate manual coverage
        self.calc_statistics(output_name='inference_statistics_xH.npz', sample_size=1000)
        self.calc_rank_coverage(sample_size=1000, output_name='rank_coverage.npz')
        self.calc_rank_coverage_per_param(sample_size=1000, output_name='rank_coverage_per_param.npz')
        
        logging.info("Inference and plotting complete.")

# ====================== Script Execution ======================

if __name__ == "__main__":
    # Define the model directory and paths
    model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR/full_EoR_pure_z12_weights_dim'
    
    redshift_values = np.array([
        5.0, 5.51724138, 6.03448276, 6.55172414, 7.06896552,
        7.5862069, 8.10344828, 8.62068966, 9.13793103, 9.65517241,
        10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
        12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
        15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931,
        17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.0
    ])
    
    
    #redshift_values = np.array([ 10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
    #   12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621 ])

    # Parameters dictionary setup
    params = {
        'n_labels': 15,         # Number of xH values
        'redshifts': redshift_values,     # Number of redshift values
        'cond_dims': 15 + 10,        # CNN output size + redshift_dim
        'plot': {
            'plot_dir': os.path.join(model_dir, 'plots_coverage'),
        }
    }

    # Load the pretrained CNN and flow models
    cnn_model = CNN()  # Use your modified CNN model
    cnn_model_path = os.path.join(model_dir, 'best_cnn_model.pth')
    if os.path.exists(cnn_model_path):
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location='cpu'))
        cnn_model.eval()
        logging.info("Loaded pretrained CNN model.")
    else:
        logging.error(f"Pretrained CNN model not found at {cnn_model_path}.")
        sys.exit(1)

    model_params = {
        'flow': {
            'n_dim': params['n_labels'],
            'n_blocks': 6,
            'n_nodes': 256,
            'cond_dims': params['cond_dims'],
            'dropout': 0.0,
            'load': False,
            'model_location': 'trained_model.pth',
            'sigmoid': False
        }
    }

    flow_model = ConditionalInvertibleBlock(model_params)
    flow_model_path = os.path.join(model_dir, 'best_flow_model.pth')
    if os.path.exists(flow_model_path):
        flow_model.flow.load_state_dict(torch.load(flow_model_path, map_location='cpu'))
        flow_model.flow.eval()
        logging.info("Loaded pretrained Flow model.")
    else:
        logging.error(f"Pretrained Flow model not found at {flow_model_path}.")
        sys.exit(1)

    # Create inference model instance
    inference_model = InferenceModel(
        params=params,
        cnn_model=cnn_model,
        flow_model=flow_model,
        data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10'],
        #data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise', 
        #   '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'],
        #data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10', 
        #'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise', 
        #'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run inference and coverage evaluations
    inference_model.main()