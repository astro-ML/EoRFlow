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
from cnn import CNN2D_pretrained as CNN  
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global as PowerSpectrumDataset
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable, Tuple
import seaborn as sns
from scipy.stats import binom

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
        self.dataset = PowerSpectrumDataset(data_dirs, exclude_unfinished_reionization=True, exclude_early_reionization=True)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Set up output directory
        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for storing CNN predictions and labels
        self.cnn_pred = None
        self.label = None
        self.n_labels = n_labels
        self.cond_dims = params['cond_dims']
        self.redshifts = params['redshifts']

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

            # Adjust redshift values as needed
            redshifts = torch.tensor(self.redshifts / 10, dtype=torch.float32).to(self.device)

            for i, (ps_batch, true_label) in enumerate(self.data_loader):
                ps_batch, true_label = ps_batch.to(self.device), true_label.to(self.device)
                #ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
                #ps_batch = ps_batch.to(device)
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # add redshifts batch

                # Forward pass through the CNN
                cnn_output = self.cnn_model(ps_batch, redshift_batch)
                cond = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, 60]

                self.cnn_pred[i] = cond.detach().cpu().numpy()
                self.label[i] = true_label.detach().cpu().numpy()

                if i % 100 == 0:
                    logging.info(f"Processed {i} samples in CNN inference.")

            np.savez(os.path.join(self.output_dir, save_name), cnn_pred=self.cnn_pred, label=self.label)

        return self.cnn_pred, self.label

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """Calculate test statistics for inferred xH values."""
        self.cnn_pred, self.label = self.find_cnn_output()
        logging.info('Calculating statistics for xH inference.')

        num_samples, num_params = self.label.shape
        mean = np.zeros((num_samples, num_params))
        lower = np.zeros((num_samples, num_params))
        upper = np.zeros((num_samples, num_params))
        rank = np.zeros((num_samples, num_params))  # Initialize rank array

        # Define confidence levels
        confidence_levels = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]  # Percentages

        # Initialize coverage counters
        coverage_counts = {level: np.zeros(num_params) for level in confidence_levels}

        for i in range(num_samples):
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            condition = torch.Tensor(self.cnn_pred[i]).unsqueeze(0).repeat(sample_size, 1).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
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

            if i % 100 == 0:
                logging.info(f"Processed {i} samples in flow inference.")

        # Calculate empirical coverage percentages
        empirical_coverage = {level: (coverage_counts[level] / num_samples) * 100 for level in confidence_levels}

        # Save statistics including rank
        np.savez(os.path.join(self.output_dir, output_name), mean=mean, lower=lower, upper=upper, label=self.label, rank=rank, coverage=empirical_coverage)
        logging.info('Statistics saved.')


    def plot_coverage(self, output_name: str = 'coverage_plot.pdf') -> None:
        """Plot coverage of predicted confidence intervals."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'), allow_pickle=True)
        coverage = data['coverage'].item()  # Retrieve the dictionary
        num_params = self.label.shape[1]
        confidence_levels = sorted(coverage.keys())  # [68, 90, 95]

        # Calculate average coverage across all parameters
        avg_coverage = {level: np.mean(coverage[level]) for level in confidence_levels}

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.plot(confidence_levels, [avg_coverage[level] for level in confidence_levels],
                marker='o', linestyle='-', color='blue', label='Empirical Coverage')
        plt.plot(confidence_levels, confidence_levels, marker='x', linestyle='--',
                color='red', label='Nominal Coverage')
        plt.xlabel('Confidence Level (%)', fontsize=15)
        plt.ylabel('Coverage (%)', fontsize=15)
        plt.title('Coverage Plot', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, output_name))
        plt.close()
        logging.info("Coverage plot saved.")


    def plot_rank_statistic(self) -> None:
        """
        Plot the rank statistic for the inferred xH values.

        This method generates rank statistic plots based on the data stored in the 'inference_statistics_xH.npz' file.
        It saves the plots as a PDF file named 'rank_statistic_xH.pdf' in the output directory. It can be used
        to detect visual biases.
        """
        logging.info('Generating rank statistic plots')
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        label = data['label']
        rank = data['rank']
        num_samples, num_params = label.shape
        bins = 15
        sample_size = int(np.max(rank))
        ranges = np.linspace(0, sample_size, bins + 1)
        avg = num_samples / bins
        low, up = binom.interval(0.99, num_samples, 1 / bins)
        size = 15  # You can adjust the fontsize as needed

        with PdfPages(os.path.join(self.output_dir, 'rank_statistic_xH.pdf')) as pdf:
            for para in range(num_params):
                plt.figure(figsize=(9, 6))
                plt.hist(rank[:, para], bins=bins, ec='teal', histtype='step')
                plt.fill_between(ranges[:-1], low, up, color='k', alpha=0.2)
                plt.plot(ranges[:-1], np.ones(bins) * avg, color='k')
                plt.title(f'Rank Statistic for xH at Redshift {np.round(self.redshifts[para],2)}', fontsize=size)
                plt.xlabel('Rank Statistic', fontsize=size)
                plt.ylabel('Frequency', fontsize=size)
                plt.xticks(fontsize=size)
                plt.yticks(fontsize=size)
                pdf.savefig()
                plt.close()
        logging.info("Rank statistic plots saved.")



    def plot_calibration(self, selected_indices: list = None) -> None:
        """Plot calibration curve comparing true vs. inferred xH values."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        label, mean, lower, upper = data['label'], data['mean'], data['lower'], data['upper']
        error_low, error_up = np.abs(mean - lower), np.abs(upper - mean)
        redshift_values = self.redshifts

        # If no indices are provided, select a few evenly spaced parameters
        if selected_indices is None:
            selected_indices = np.linspace(0, label.shape[1] - 1, num=5, dtype=int)

        with PdfPages(os.path.join(self.output_dir, 'calibration_xH.pdf')) as pdf:
            for idx in selected_indices:
                plt.figure(figsize=(8, 6))
                plt.errorbar(label[:, idx], mean[:, idx], yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='.', color='blue', ecolor='lightblue', elinewidth=1, capsize=2)
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlabel('True Value')
                plt.ylabel(f'Inferred xH at z={np.round(redshift_values[idx],2)}')
                plt.title(f'Calibration Plot for xH at z={np.round(redshift_values[idx],2)}', fontsize=14)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(True)
                pdf.savefig()
                plt.close()
        logging.info("Calibration plot saved.")

    def plot_reionization_history(self, index: int) -> None:
        """Plot the reionization history with uncertainty bands for a single test sample."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        mean = data['mean'][index]
        lower = data['lower'][index]
        upper = data['upper'][index]
        true_values = data['label'][index]

        redshift_values = np.linspace(5.0, 20.0, num=mean.shape[0])  # Adjust based on your redshift range

        plt.figure(figsize=(10, 6))
        plt.fill_between(redshift_values, lower, upper, color='lightblue', alpha=0.5, label='68% Confidence Interval')
        plt.plot(redshift_values, mean, color='blue', label='Inferred Mean')
        plt.plot(redshift_values, true_values, color='red', linestyle='--', label='True xH')
        plt.xlabel('Redshift z')
        plt.ylabel('Neutral Hydrogen Fraction xH')
        plt.title(f'Reionization History for Sample {index + 1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'reionization_history_sample_{index + 1}.pdf'))
        plt.close()
        logging.info(f"Reionization history plot saved for sample {index + 1}.")

    def plot_error_vs_redshift(self) -> None:
        """Plot the mean absolute error of inferred xH values over redshift."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        label, mean = data['label'], data['mean']
        mae = np.mean(np.abs(mean - label), axis=0)
        redshift_values = np.linspace(5.0, 20.0, num=mae.shape[0])  # Adjust as needed

        plt.figure(figsize=(10, 6))
        plt.plot(redshift_values, mae, marker='o')
        plt.xlabel('Redshift z')
        plt.ylabel('Mean Absolute Error of xH')
        plt.title('Mean Absolute Error of Inferred xH Over Redshift')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'mae_vs_redshift.pdf'))
        plt.close()
        logging.info("MAE vs. redshift plot saved.")

    def plot_correlation_heatmap(self) -> None:
        """Plot a correlation heatmap of the inferred xH parameters."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        mean = data['mean']
        correlation_matrix = np.corrcoef(mean, rowvar=False)

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
        plt.title('Correlation Heatmap of Inferred xH Parameters')
        plt.xlabel('Parameter Index')
        plt.ylabel('Parameter Index')
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap_xH.pdf'))
        plt.close()
        logging.info("Correlation heatmap saved.")

    def plot_sampled_values_histogram(self, sample_size: int = 1000) -> None:
        """
        Plot a histogram of all sampled xH values to check if they are between 0 and 1.
        """
        logging.info('Generating histogram of all sampled xH values.')

        # Ensure that the statistics have been calculated
        if not os.path.exists(os.path.join(self.output_dir, 'all_samples_xH.npy')):
            logging.info('Aggregating samples from flow model.')
            self.cnn_pred, self.label = self.find_cnn_output()

            num_samples = len(self.cnn_pred)
            all_samples = []

            for i in range(num_samples):
                z = torch.randn((sample_size, self.n_labels)).to(self.device)
                condition = torch.tensor(self.cnn_pred[i], dtype=torch.float32).unsqueeze(0).repeat(sample_size, 1).to(self.device)
                samples, _ = self.flow_model(z, c=[condition], rev=True)
                samples = samples.detach().cpu().numpy()
                all_samples.append(samples)

                if i % 100 == 0:
                    logging.info(f"Processed {i}/{num_samples} samples.")

            # Concatenate all samples into a single array
            all_samples = np.concatenate(all_samples, axis=0)
            np.save(os.path.join(self.output_dir, 'all_samples_xH.npy'), all_samples)
        else:
            logging.info('Loading aggregated samples from file.')
            all_samples = np.load(os.path.join(self.output_dir, 'all_samples_xH.npy'))

        # Flatten the array to 1D
        all_samples_flat = all_samples.flatten()

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(all_samples_flat, bins=100, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel('Sampled xH Values', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Histogram of All Sampled xH Values', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'sampled_values_histogram_xH.pdf'))
        plt.close()
        logging.info('Histogram of all sampled xH values saved.')

    def main(self) -> None:
        """Main function to run inference and create plots."""
        self.calc_statistics(output_name='inference_statistics_xH.npz')
        indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
        self.plot_calibration(selected_indices=indices)
        for i in range(20):  # Generate plots for the first 5 samples as an example
            self.plot_reionization_history(index=i)
        self.plot_error_vs_redshift()
        #self.plot_correlation_heatmap()
        self.plot_rank_statistic()  
        self.plot_sampled_values_histogram()
        self.plot_coverage()
        logging.info("Inference and plotting complete.")

# Define the model directory and paths
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pretrained_noise'  # Update to your output directory

redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
        7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
       10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
       12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
       15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
       17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])

# Parameters dictionary setup
params = {
    'n_labels': 30,         # Number of xH values
    'cond_dims': 60,        # CNN output size (since we didn't add redshift in the conditioning)
    'redshifts': redshift_values,     # Number of redshift values
    'plot': {
        'plot_dir': os.path.join(model_dir, 'plots'),
    }
}

# Load the fine-tuned CNN and flow models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load(os.path.join(model_dir, 'finetuned_cnn_model.pth'), map_location=device))
cnn_model.eval()


model_params = {
    'flow': {
        'n_dim': params['n_labels'],
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': params['cond_dims'],
        'dropout': 0.3,
        'load': False,
        'model_location': 'trained_model.pth',
    }
}

flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)
flow_model.flow.load_state_dict(torch.load(os.path.join(model_dir, 'finetuned_flow_model.pth'), map_location=device))
flow_model.flow.eval()

# Create inference model instance
inference_model = InferenceModel(
    params=params,
    cnn_model=cnn_model,
    flow_model=flow_model,
    data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise',
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'],
    device=device
)

# Set up logging
logging.basicConfig(
    filename=os.path.join(inference_model.output_dir, 'inference.log'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Run inference
inference_model.main()
