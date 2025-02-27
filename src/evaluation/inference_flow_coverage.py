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
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset
from matplotlib.backends.backend_pdf import PdfPages
from getdist import plots, MCSamples

class InferenceModel:
    """
    A class for inferring the xH values using a flow model with 2DPS as input.
    Updated to include a line plot for log-probability (rank-based) coverage.
    
    The flow takes as condition the flattened 2D power spectrum (e.g. a (3,10,10) array
    flattened to 300 elements) and infers three xH values.
    """
    def __init__(self, params: dict, flow_model: nn.Module, data_dir: str, device: str = 'cpu') -> None:
        self.params = params['plot']
        # We assume that flow_model.flow is the invertible network.
        self.flow_model = flow_model.flow.to(device)
        self.device = device

        # Initialize dataset and data loader.
        self.dataset = PowerSpectrumDataset(data_dir)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Set up output directory.
        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for true labels.
        self.label = None

    def get_data(self) -> np.ndarray:
        """Load test data and labels."""
        num_samples = len(self.dataset)
        self.label = np.zeros((num_samples, 3))  # assuming 3 xH values
        for i, (ps_data, true_label) in enumerate(self.data_loader):
            self.label[i] = true_label.detach().cpu().numpy()
        return self.label

    def compute_log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of x under the flow model given condition c.
        Uses the forward pass (rev=False) to compute the latent variable z and the log-det.
        """
        # Ensure that c is batched.
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        z, log_det = self.flow_model(x, c=[c], rev=False)
        base_log_prob = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1)
        return base_log_prob + log_det

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """Calculate test statistics (mean, 16th/84th percentiles) for inferred xH values."""
        self.label = self.get_data()
        logging.info('Calculating statistics for xH inference.')
        mean = np.zeros_like(self.label)
        lower = np.zeros_like(self.label)
        upper = np.zeros_like(self.label)

        for i, ps_data in enumerate(self.data_loader):
            ps_data = ps_data[0].to(self.device)  # extract the 2DPS data
            z = torch.randn((sample_size, 3)).to(self.device)
            # Sample from the flow (using rev=True).
            samples, _ = self.flow_model(z, c=[ps_data.repeat(sample_size, 1).to(self.device)], rev=True)
            samples = samples.detach().cpu().numpy()

            mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])
            stats = mc_samples.getMargeStats()
            for j, name in enumerate(['xH1', 'xH2', 'xH3']):
                mean[i, j] = stats.parWithName(name).mean
                lower[i, j] = stats.parWithName(name).limits[0].lower
                upper[i, j] = stats.parWithName(name).limits[0].upper

            if i % 100 == 0:
                logging.info(f"Processed {i} samples for statistics.")
        np.savez(os.path.join(self.output_dir, output_name), mean=mean, lower=lower, upper=upper, label=self.label)
        logging.info('Statistics saved.')

    def plot_calibration(self) -> None:
        """Plot calibration curves comparing true vs. inferred xH values."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        label, mean, lower, upper = data['label'], data['mean'], data['lower'], data['upper']
        error_low, error_up = np.abs(mean - lower), np.abs(upper - mean)

        with PdfPages(os.path.join(self.output_dir, 'calibration_xH.pdf')) as pdf:
            for idx, param in enumerate(['xH1', 'xH2', 'xH3']):
                plt.figure(figsize=(8, 6))
                plt.errorbar(label[:, idx], mean[:, idx],
                             yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='.', color='blue', ecolor='lightblue',
                             elinewidth=1, capsize=2)
                plt.plot(label[:, idx], label[:, idx], color='red', linestyle='--')
                plt.xlabel('True Value')
                plt.ylabel(f'Inferred {param}')
                plt.title(f'Calibration plot for {param}')
                pdf.savefig()
                plt.close()
        logging.info("Calibration plot saved.")

    def plot_corner(self, index: int, sample_size: int = 1000) -> None:
        """Generate a corner plot for the posterior of a single test sample."""
        z = torch.randn((sample_size, 3)).to(self.device)
        ps_data, true_label = self.dataset[index]
        ps_data = ps_data.to(self.device)
        samples, _ = self.flow_model(z, c=[ps_data.repeat(sample_size, 1).to(self.device)], rev=True)
        samples = samples.detach().cpu().numpy()
        mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])

        g = plots.get_subplot_plotter()
        g.triangle_plot([mc_samples], filled=True)

        true_values = true_label.numpy()
        for i, name in enumerate(['xH1', 'xH2', 'xH3']):
            g.subplots[i, i].axvline(true_values[i], color='red', linestyle='--')

        for i in range(2):
            for j in range(i + 1, 3):
                ax = g.subplots[j, i]
                ax.plot(true_values[i], true_values[j], 'ro', markersize=6)

        plot_path = os.path.join(self.output_dir, f'corner_plot_xH_sample_{index + 1}.pdf')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Corner plot saved for sample {index + 1}.")

    def plot_log_prob_coverage_line(self, sample_size: int = 1000, num_bins: int = 50) -> None:
        """
        Compute and plot rank-based coverage as a line plot.
        
        For each test sample, the true xH log-probability is computed and compared with the log-probabilities
        of posterior samples drawn from the flow (using the same condition). The rank (fraction of posterior
        samples with lower log-probability than the true value) is computed. Then we plot the empirical CDF of
        (1 - rank) versus the ideal uniform CDF.
        """
        ranks = []
        for i, (ps_data, true_label) in enumerate(self.data_loader):
            ps_data = ps_data.to(self.device)
            # true_label has shape [1, n_dim]
            true_label = true_label.to(self.device).view(1, -1)
            true_log_prob = self.compute_log_prob(true_label, ps_data)
            
            z = torch.randn((sample_size, 3)).to(self.device)
            cond = ps_data.repeat(sample_size, 1)
            samples, _ = self.flow_model(z, c=[cond], rev=True)
            sample_log_probs = self.compute_log_prob(samples, cond)
            
            # Compute the fraction of samples whose log-prob is lower than the true log-prob.
            rank = (sample_log_probs < true_log_prob).float().mean().item()
            ranks.append(rank)
            
            if i % 100 == 0:
                logging.info(f"Processed {i} samples for log prob coverage (line plot).")
        ranks = np.array(ranks)  # shape: (num_samples,)
        
        # We now compute the empirical CDF of (1 - rank). In an ideal case (perfect calibration)
        # (1 - rank) would be uniformly distributed on [0, 1].
        x_values = np.linspace(0, 1, num_bins)
        quantiles = np.quantile(1 - ranks, x_values)
        
        plt.figure(figsize=(5, 5))
        plt.plot(quantiles, x_values, label='Empirical coverage', alpha=0.9)
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal coverage')
        plt.xlabel('Quantile of (1 - rank)')
        plt.ylabel('Cumulative probability')
        plt.legend()
        plt.title('Rank-based Coverage using Log Probabilities')
        plot_path = os.path.join(self.output_dir, 'rank_coverage_line_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Rank coverage line plot saved to {plot_path}.")

    def plot_individual_ranks(self, sample_size: int = 1000, bins: int = 20) -> None:
        """
        Plot histograms of the ranks of true parameters among posterior samples for each xH dimension.
        Normalizing the rank (dividing by sample_size) gives values between 0 and 1.
        """
        ranks = {'xH1': [], 'xH2': [], 'xH3': []}
        param_names = ['xH1', 'xH2', 'xH3']
        for i, (ps_data, true_label) in enumerate(self.data_loader):
            ps_data = ps_data.to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            z = torch.randn((sample_size, 3)).to(self.device)
            cond = ps_data.repeat(sample_size, 1)
            samples, _ = self.flow_model(z, c=[cond], rev=True)
            samples_np = samples.detach().cpu().numpy()
            true_np = true_label.detach().cpu().numpy().flatten()
            for j in range(3):
                rank_val = np.sum(samples_np[:, j] < true_np[j])
                ranks[param_names[j]].append(rank_val)
            if i % 100 == 0:
                logging.info(f"Processed {i} samples for individual rank plots.")
        for key in ranks:
            ranks[key] = np.array(ranks[key])
        with PdfPages(os.path.join(self.output_dir, 'individual_ranks.pdf')) as pdf:
            for idx, param in enumerate(param_names):
                plt.figure(figsize=(8, 6))
                plt.hist(ranks[param] / sample_size, bins=bins, density=True, alpha=0.7,
                         color='green', edgecolor='black')
                plt.xlabel('Normalized Rank')
                plt.ylabel('Density')
                plt.title(f'Rank Histogram for {param}')
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        logging.info("Individual rank plots saved.")

    def main(self) -> None:
        """Run inference and generate all plots."""
        self.calc_statistics(output_name='inference_statistics_xH.npz')
        self.plot_calibration()
        self.plot_log_prob_coverage_line()
        self.plot_individual_ranks()
        for i in range(5):  # For example, generate corner plots for the first 5 samples.
            self.plot_corner(index=i)
        logging.info("Inference and plotting complete.")


# =====================================================
# Script execution
# =====================================================

# Define the model directory and paths.
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/flow_1'

# Parameters dictionary setup.
params = {
    'plot': {
        'plot_dir': os.path.join(model_dir, 'plots'),
        'parameters': [
            ["xH1", 0, 1, r"$x_{H1}$"],
            ["xH2", 0, 1, r"$x_{H2}$"],
            ["xH3", 0, 1, r"$x_{H3}$"]
        ]
    }
}

# Flow model parameters.
model_params = {
    'flow': {
        'n_dim': 3,
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': 303,  # Adjust based on the flattened 2DPS input size.
        'dropout': 0.0,
        'load': False,
        'model_location': 'trained_model.pth',
    }
}

flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.load_state_dict(torch.load(os.path.join(model_dir, 'trained_model.pth')))
flow_model.flow.eval()

# Create the inference model instance.
inference_model = InferenceModel(
    params=params,
    flow_model=flow_model,
    data_dir=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Run inference and generate plots.
inference_model.main()