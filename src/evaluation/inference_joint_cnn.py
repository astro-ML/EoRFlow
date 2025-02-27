import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
from cnn import CNN3D_SKA as CNN  
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_SKA
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import binom
from getdist import plots, MCSamples

class InferenceModel:
    """
    InferenceModel for joint CNN + flow inference.
    
    This class:
      - Computes the CNN output (or loads it) which is concatenated with redshifts to form a condition.
      - Uses the flow to sample posterior xH values.
      - Computes posterior statistics and produces:
          * Calibration scatter plots.
          * Corner (contour) plots.
          * Individual rank histograms.
          * Log-probability coverage line plot.
    """
    def __init__(self, params: dict, cnn_model: nn.Module, flow_model: nn.Module, data_dir: list, device: str = 'cpu') -> None:
        self.params = params['plot']
        self.cnn_model = cnn_model.to(device)
        self.flow_model = flow_model.flow.to(device)
        self.device = device

        self.n_labels = params['n_labels']      # e.g., 3 xH values
        self.redshifts = params['redshifts']      # e.g., [6.54, 7.19, 7.96]
        self.cond_dims = params['cond_dims']      # e.g., 13 = CNN output (10) + 3 redshifts

        # Initialize dataset and data loader.
        self.dataset = PowerSpectrumDataset_SKA(data_dir)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Setup output directory.
        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholders for storing CNN conditions and true labels.
        self.cnn_pred = None
        self.label = None

    def find_cnn_output(self, save_name: str = 'cnn_output_xH.npz'):
        """Run the CNN on test data (or load from file) to get predictions.
           The CNN output is concatenated with the redshifts to form the condition.
        """
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

            # Normalize redshifts (example: divide by 10)
            z1, z2, z3 = self.redshifts
            redshifts = torch.tensor([z1/10, z2/10, z3/10], dtype=torch.float32).to(self.device)

            for i, (ps_data, true_label) in enumerate(self.data_loader):
                ps_data, true_label = ps_data.to(self.device), true_label.to(self.device)
                # For a 3D CNN, ensure the input has the required dimensions.
                ps_data = ps_data.unsqueeze(1)  # add channel dimension if needed
                redshift_batch = redshifts.repeat(ps_data.size(0), 1)  # shape: [batch_size, 3]
                cnn_output = self.cnn_model(ps_data, redshift_batch)
                # Concatenate CNN output with redshifts to form the full condition.
                pred = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, cond_dims]
                self.cnn_pred[i] = pred.detach().cpu().numpy()
                self.label[i] = true_label.detach().cpu().numpy()

            np.savez(os.path.join(self.output_dir, save_name), cnn_pred=self.cnn_pred, label=self.label)
        return self.cnn_pred, self.label

    def compute_log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-probability of x under the flow model given condition cond.
        """
        z, log_det = self.flow_model(x, c=[cond], rev=False)
        base_log_prob = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1)
        return base_log_prob + log_det

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000):
        """
        For each test sample, use the stored CNN condition to sample from the flow,
        then compute posterior statistics (mean, 16th/84th percentiles), rank statistics,
        and empirical coverage.
        """
        self.cnn_pred, self.label = self.find_cnn_output()
        logging.info('Calculating statistics for xH inference.')

        num_samples, num_params = self.label.shape
        mean = np.zeros((num_samples, num_params))
        lower = np.zeros((num_samples, num_params))
        upper = np.zeros((num_samples, num_params))
        rank = np.zeros((num_samples, num_params))

        for i in range(num_samples):
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            condition = torch.tensor(self.cnn_pred[i], dtype=torch.float32).unsqueeze(0).repeat(sample_size, 1).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            samples = samples.detach().cpu().numpy()

            # clip for diagnostics
            samples = np.clip(samples, 0, 1)

            # Compute statistics for each parameter.
            mean[i] = np.mean(samples, axis=0)
            lower[i] = np.percentile(samples, 16, axis=0)
            upper[i] = np.percentile(samples, 84, axis=0)

            # Compute rank: count number of posterior samples below the true value.
            for j in range(num_params):
                rank[i, j] = np.sum(samples[:, j] < self.label[i, j])

            if i % 100 == 0:
                logging.info(f"Processed {i} samples for statistics.")

        np.savez(os.path.join(self.output_dir, output_name),
                 mean=mean, lower=lower, upper=upper, label=self.label, rank=rank)
        logging.info('Statistics saved with coverage.')

    def plot_calibration(self, selected_indices: list = None):
        """
        Plot a calibration scatter plot (true vs. inferred xH values) with error bars.
        """
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        label, mean, lower, upper = data['label'], data['mean'], data['lower'], data['upper']
        error_low = np.abs(mean - lower)
        error_up = np.abs(upper - mean)

        if selected_indices is None:
            selected_indices = list(range(self.n_labels))
        param_names = ['xH1', 'xH2', 'xH3']  # Adjust as needed

        with PdfPages(os.path.join(self.output_dir, 'calibration_scatter.pdf')) as pdf:
            for idx in selected_indices:
                plt.figure(figsize=(8, 6))
                plt.errorbar(label[:, idx], mean[:, idx],
                             yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='o', color='blue', ecolor='lightblue', capsize=3,
                             label='Inferred')
                plt.plot(label[:, idx], label[:, idx], 'r--', label='Ideal')
                plt.xlabel('True Value', fontsize=15)
                plt.ylabel(f'Inferred {param_names[idx]}', fontsize=15)
                plt.title(f'Calibration Scatter for {param_names[idx]}', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True)
                pdf.savefig()
                plt.close()
        logging.info("Calibration scatter plot saved.")

    def plot_corner(self, index: int, sample_size: int = 1000) -> None:
        """
        Generate a corner (contour) plot for the posterior of a single test sample.
        """
        condition = torch.tensor(self.cnn_pred[index], dtype=torch.float32).unsqueeze(0).to(self.device)
        z = torch.randn((sample_size, self.n_labels)).to(self.device)
        samples, _ = self.flow_model(z, c=[condition.repeat(sample_size, 1)], rev=True)
        samples = samples.detach().cpu().numpy()

        # clip for diagnostics
        samples = np.clip(samples, 0, 1)
        
        mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])

        g = plots.get_subplot_plotter()
        g.triangle_plot([mc_samples], filled=True)

        true_values = self.label[index]
        for i, name in enumerate(['xH1', 'xH2', 'xH3']):
            g.subplots[i, i].axvline(true_values[i], color='red', linestyle='--')
        for i in range(2):
            for j in range(i+1, 3):
                ax = g.subplots[j, i]
                ax.plot(true_values[i], true_values[j], 'ro', markersize=6)

        plot_path = os.path.join(self.output_dir, f'corner_plot_sample_{index+1}.pdf')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Corner plot saved for sample {index+1}.")

    def plot_log_prob_coverage_line(self, sample_size: int = 1000) -> None:
        """
        Compute and plot a log-probability coverage line.
        
        For each test sample, compute the log-probability of the true xH values versus
        that of posterior samples (using the stored condition). Then compute the fraction of
        posterior samples with lower log-probability than the true value (the rank) and plot
        the empirical CDF of (1 - rank) against the ideal uniform CDF.
        """
        ranks = []
        if self.cnn_pred is None or self.label is None:
            self.find_cnn_output()

        for i, (ps_data, true_label) in enumerate(self.data_loader):
            # Use stored condition from CNN output.
            condition = torch.tensor(self.cnn_pred[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            true_log_prob = self.compute_log_prob(true_label, condition)
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            cond_rep = condition.repeat(sample_size, 1)
            samples, _ = self.flow_model(z, c=[cond_rep], rev=True)

            # clip for diagnostics
            samples = torch.clamp(samples, min=0.0, max=1.0)

            sample_log_probs = self.compute_log_prob(samples, cond_rep)
            rank_val = (sample_log_probs < true_log_prob).float().mean().item()
            ranks.append(rank_val)

            if i % 100 == 0:
                logging.info(f"Processed {i} samples for log-prob coverage.")

        ranks = np.array(ranks)
        sorted_vals = np.sort(1 - ranks)
        cdf = np.linspace(0, 1, len(sorted_vals))
        
        plt.figure(figsize=(6, 6))
        plt.plot(sorted_vals, cdf, label='Empirical CDF')
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal Uniform')
        plt.xlabel('Quantile of (1 - rank)', fontsize=15)
        plt.ylabel('Cumulative Probability', fontsize=15)
        plt.title('Log-Probability Coverage', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'log_prob_coverage_line.pdf'))
        plt.close()
        logging.info("Log-probability coverage line plot saved.")

    def plot_individual_ranks(self, sample_size: int = 1000, bins: int = 20) -> None:
        """
        Plot histograms of the individual parameter ranks (normalized by sample_size) over the test set.
        
        For each test sample, using the stored CNN condition, draw posterior samples from the flow,
        count (for each xH parameter) how many samples are below the true value, normalize by the sample_size,
        and plot the resulting distribution as a histogram.
        """
        ranks = {'xH1': [], 'xH2': [], 'xH3': []}
        param_names = ['xH1', 'xH2', 'xH3']
        if self.cnn_pred is None or self.label is None:
            self.find_cnn_output()

        for i, (ps_data, true_label) in enumerate(self.data_loader):
            # Use stored CNN condition.
            condition = torch.tensor(self.cnn_pred[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            samples, _ = self.flow_model(z, c=[condition.repeat(sample_size, 1)], rev=True)

            # clip for diagnostics
            samples = torch.clamp(samples, min=0.0, max=1.0)

            samples_np = samples.detach().cpu().numpy()
           

            true_np = true_label.detach().cpu().numpy().flatten()
            for j in range(self.n_labels):
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
                plt.xlabel('Normalized Rank', fontsize=15)
                plt.ylabel('Density', fontsize=15)
                plt.title(f'Rank Histogram for {param}', fontsize=16)
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        logging.info("Individual rank plots saved.")

    def plot_predicted_histogram(self, sample_size: int = 1000) -> None:
        """
        For each test sample, use the stored CNN condition to sample from the flow,
        compute the posterior mean prediction for xH, and then plot a histogram of these
        predicted xH values. Additionally, log the number of predictions below 0 and above 1.
        """
        if self.cnn_pred is None or self.label is None:
            self.find_cnn_output()

        num_samples = len(self.dataset)
        predicted_means = []  # shape: (num_samples, n_labels)
        
        for i in range(num_samples):
            # Get the stored condition for sample i.
            condition = torch.tensor(self.cnn_pred[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            # Sample from the base distribution.
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            # Generate posterior samples.
            samples, _ = self.flow_model(z, c=[condition.repeat(sample_size, 1)], rev=True)
            samples = samples.detach().cpu().numpy()

            # clip for diagnostics
            samples = np.clip(samples, 0, 1)

            # Compute the mean predicted xH for sample i.
            sample_mean = np.mean(samples, axis=0)
            predicted_means.append(sample_mean)
        
        predicted_means = np.array(predicted_means)  # shape: (num_samples, n_labels)
        
        # Plot a histogram for each parameter.
        for j in range(self.n_labels):
            values = predicted_means[:, j]
            logging.info("Min predicted value:", np.min(predicted_means[:, j]))
            logging.info("Max predicted value:", np.max(predicted_means[:, j]))
            plt.figure(figsize=(8, 6))
            plt.hist(values, bins=100, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel(f"Predicted xH{j+1}")
            plt.xlim(-0.2,1)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Predicted xH{j+1} Values")
            plot_path = os.path.join(self.output_dir, f'predicted_xH{j+1}_histogram.pdf')
            plt.savefig(plot_path)
            plt.close()
            # Count and log values outside [0, 1].
            num_below = np.sum(values < 0)
            num_above = np.sum(values > 1)
            logging.info(f"For xH{j+1}: {num_below} values below 0, {num_above} values above 1 out of {len(values)} samples.")
            

    def main(self) -> None:
        """Run the full inference pipeline and generate all plots."""
        logging.info("Starting inference and plotting...")
        self.calc_statistics(output_name='inference_statistics_xH.npz')
        self.plot_calibration()
        self.plot_log_prob_coverage_line()
        self.plot_individual_ranks()
        # For example, generate corner plots for the first 20 samples.
        for i in range(min(5, len(self.dataset))):
            self.plot_corner(index=i)
        self.plot_predicted_histogram(sample_size=1000)
        logging.info("Inference and plotting complete.")

if __name__ == '__main__':
    # Setup logging to the console.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define the model directory and parameters.
    model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/SKA_CNN/CNN_6_256_weighed_BN_9'
    z1, z2, z3 = 6.54, 7.19, 7.96
    redshift_values = np.array([z1, z2, z3])
    params = {
        'n_labels': 3,
        'redshifts': redshift_values,
        'cond_dims': 3 + 9,
        'plot': {
            'plot_dir': os.path.join(model_dir, 'plots'),
        }
    }
    
    # Load pretrained CNN and flow models.
    cnn_model = CNN()  # Ensure the CNN outputs the correct dimension.
    cnn_model.load_state_dict(torch.load(os.path.join(model_dir, 'cnn_model.pth')))
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
    flow_model.flow.load_state_dict(torch.load(os.path.join(model_dir, 'flow_model.pth')))
    flow_model.flow.eval()
    
    data_dirs = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    inference_model = InferenceModel(
        params=params,
        cnn_model=cnn_model,
        flow_model=flow_model,
        data_dir=data_dirs,
        device=device
    )
    
    # Run inference and generate all plots.
    inference_model.main()