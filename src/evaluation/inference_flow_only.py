import os
import sys
sys.path.append('/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/models')
sys.path.append('/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/data_tools')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset
from matplotlib.backends.backend_pdf import PdfPages
from getdist import plots, MCSamples

# Define output directory
output_dir = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/noise_augmented/Pk_window_10_512_std5_Gaussian0.01'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Set up logging
log_filename = os.path.join(output_dir, 'inference.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # Overwrites the log file if it exists
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)  # Change to DEBUG for more detailed logs if needed
logging.getLogger('getdist').setLevel(logging.WARNING)
logging.getLogger('getdist.mcsamples').setLevel(logging.WARNING)

class InferenceModel:
    """
    InferenceModel class for performing inference with a trained flow model.
    
    This class produces:
      - Calibration scatter plots: true vs. inferred xH values (with error bars).
      - Corner plots: 2D posterior contours for selected test samples.
      - Individual rank histograms: distribution of the rank of the true value among posterior samples.
      - Log-probability coverage line plot: comparing the empirical CDF of (1 - rank) against the ideal uniform CDF.
    
    Args:
        params (dict): Dictionary with plotting parameters (e.g. output directory, parameter names).
        flow_model (nn.Module): Trained flow model. It is expected to have a `.flow` attribute.
        data_dir (str or list): Path(s) to the test dataset.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    def __init__(self, params: dict, flow_model: torch.nn.Module, data_dir, device: str = 'cpu', sigmoid=False, temperature=1):
        self.params = params.get('plot', {})
        self.flow_model = flow_model.flow.to(device)
        self.device = device
        self.sigmoid = sigmoid
        self.temperature = temperature
        # Initialize dataset and data loader.
        self.dataset = PowerSpectrumDataset(data_dir, add_noise=True, augment_noise=True, add_gaussian=True, gaussian_std=0.01, std_strength=5.0, k_scale=True, convert_dimensionless=False, logit=False)
        
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Setup output directory.
        self.output_dir = self.params.get('plot_dir', './plots')
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for true labels (will be filled in get_data)
        self.labels = None

    def get_data(self) -> np.ndarray:
        """
        Loads the true labels from the dataset.
        
        Returns:
            np.ndarray: Array of true labels with shape (num_samples, 3).
        """
        num_samples = len(self.dataset)
        self.labels = np.zeros((num_samples, 3))  # assuming 3 xH values
        for i, (_, label) in enumerate(self.data_loader):
            self.labels[i] = label.detach().cpu().numpy().flatten()
        return self.labels

    def calc_statistics(self, output_filename: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """
        Calculate Monte Carlo statistics (mean, 16th and 84th percentiles) for the posterior of each sample.
        Saves the results (with the true labels) to a file.
        
        Args:
            output_filename (str): Filename for saving the statistics.
            sample_size (int): Number of samples drawn from the flow for each test sample.
        """
        self.get_data()
        num_samples = len(self.dataset)
        mean = np.zeros((num_samples, 3))
        lower = np.zeros((num_samples, 3))
        upper = np.zeros((num_samples, 3))
        
        for i, (ps_data, _) in enumerate(self.data_loader):
            ps_data = ps_data.to(self.device)
            cond = ps_data.repeat(sample_size, 1)
            z = torch.randn((sample_size, 3)).to(self.device) * np.sqrt(self.temperature)
            samples, _ = self.flow_model(z, c=[cond], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            #samples = torch.clamp(samples, 0, 1)
        
            samples = samples.detach().cpu().numpy()
            
            # Use getdist's MCSamples to calculate statistics
            mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])
            stats = mc_samples.getMargeStats()
            for j, name in enumerate(['xH1', 'xH2', 'xH3']):
                mean[i, j] = stats.parWithName(name).mean
                lower[i, j] = stats.parWithName(name).limits[0].lower
                upper[i, j] = stats.parWithName(name).limits[0].upper
            
            if i % 100 == 0:
                logging.info(f"Processed {i} samples for statistics.")
        
        np.savez(os.path.join(self.output_dir, output_filename), mean=mean, lower=lower, upper=upper, label=self.labels)
        logging.info("Inference statistics saved.")

    def plot_calibration_scatter(self, stats_filename: str = 'inference_statistics_xH.npz') -> None:
        """
        Plot a calibration scatter plot comparing the true values with the inferred posterior means,
        with error bars showing the 16th and 84th percentile ranges.
        
        Args:
            stats_filename (str): Filename of the saved statistics (from calc_statistics).
        """
        data = np.load(os.path.join(self.output_dir, stats_filename))
        labels = data['label']
        mean = data['mean']
        lower = data['lower']
        upper = data['upper']
        error_low = np.abs(mean - lower)
        error_up = np.abs(upper - mean)
        
        with PdfPages(os.path.join(self.output_dir, 'calibration_scatter.pdf')) as pdf:
            for idx, param in enumerate(['xH1', 'xH2', 'xH3']):
                plt.figure(figsize=(8, 6))
                plt.errorbar(labels[:, idx], mean[:, idx],
                             yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='o', color='blue', ecolor='lightblue', capsize=3,
                             label='Inferred')
                plt.plot(labels[:, idx], labels[:, idx], 'r--', label='Ideal')
                plt.xlabel('True Value')
                plt.ylabel(f'Inferred {param}')
                plt.title(f'Calibration Scatter Plot for {param}')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
        logging.info("Calibration scatter plot saved.")

    def plot_corner(self, sample_indices: list = None, sample_size: int = 1000) -> None:
        """
        Generate corner plots (using getdist) for selected test samples.
        
        Args:
            sample_indices (list): List of indices for which to generate corner plots.
                                   If None, defaults to the first 5 samples.
            sample_size (int): Number of posterior samples for each plot.
        """
        if sample_indices is None:
            sample_indices = list(range(5))
        
        for idx in sample_indices:
            ps_data, true_label = self.dataset[idx]
            ps_data = ps_data.to(self.device)
            cond = ps_data.repeat(sample_size, 1)
            z = torch.randn((sample_size, 3)).to(self.device)
            samples, _ = self.flow_model(z, c=[cond], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            #samples = torch.clamp(samples, 0, 1)
            samples = samples.detach().cpu().numpy()
            mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])
            
            g = plots.get_subplot_plotter()
            g.triangle_plot([mc_samples], filled=True)
            
            # Mark the true values
            true_values = true_label.detach().cpu().numpy().flatten()
            for i, name in enumerate(['xH1', 'xH2', 'xH3']):
                g.subplots[i, i].axvline(true_values[i], color='red', linestyle='--')
            for i in range(2):
                for j in range(i + 1, 3):
                    ax = g.subplots[j, i]
                    ax.plot(true_values[i], true_values[j], 'ro', markersize=6)
            
            plot_path = os.path.join(self.output_dir, f'corner_plot_sample_{idx+1}.pdf')
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Corner plot saved for sample {idx+1}.")

    def compute_log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-probability of x under the flow model given condition c.
        
        Args:
            x (torch.Tensor): Input data (or posterior samples).
            c (torch.Tensor): Conditioning information.
        
        Returns:
            torch.Tensor: Log-probability for each sample.
        """
        if c.dim() == 1:
            c = c.unsqueeze(0)
        z, log_det = self.flow_model(x, c=[c], rev=False)
        base_log_prob = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1)
        return base_log_prob + log_det

    def plot_log_prob_coverage_line(self, sample_size: int = 1000) -> None:
        """
        Compute and plot a log-probability coverage line.
        
        For each test sample, the log-probability of the true xH value is computed
        and compared to that of posterior samples. The rank (i.e. fraction of posterior
        samples with lower log-probability) is calculated and the empirical CDF of (1 - rank)
        is plotted against the ideal uniform CDF.
        
        Args:
            sample_size (int): Number of posterior samples.
        """
        ranks = []
        for i, (ps_data, true_label) in enumerate(self.data_loader):
            ps_data = ps_data.to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            true_log_prob = self.compute_log_prob(true_label, ps_data)

            z = torch.randn((sample_size, 3)).to(self.device)
            cond = ps_data.repeat(sample_size, 1)
            samples, _ = self.flow_model(z, c=[cond], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            #samples = torch.clamp(samples, 0, 1)
            sample_log_probs = self.compute_log_prob(samples, cond)

            rank = (sample_log_probs < true_log_prob).float().mean().item()
            ranks.append(rank)

            if i % 100 == 0:
                logging.info(f"Processed {i} samples for log-prob coverage.")

        ranks = np.array(ranks)
        sorted_vals = np.sort(1 - ranks)
        cdf = np.linspace(0, 1, len(sorted_vals))
        
        plt.figure(figsize=(6, 6))
        plt.plot(sorted_vals, cdf, label='Empirical CDF')
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal Uniform')
        plt.xlabel('Quantile of (1 - rank)')
        plt.ylabel('Cumulative Probability')
        plt.title('Log-Probability Coverage')
        plt.legend()
        plot_path = os.path.join(self.output_dir, 'log_prob_coverage_line.pdf')
        plt.savefig(plot_path)
        plt.close()
        logging.info("Log-probability coverage line plot saved.")

    def plot_individual_ranks(self, sample_size: int = 1000, bins: int = 20) -> None:
        """
        Plot histograms of the individual parameter ranks (normalized by sample_size) over the test set.
        
        Args:
            sample_size (int): Number of posterior samples per test sample.
            bins (int): Number of bins in the histogram.
        """
        ranks = {'xH1': [], 'xH2': [], 'xH3': []}
        param_names = ['xH1', 'xH2', 'xH3']
        for i, (ps_data, true_label) in enumerate(self.data_loader):
            ps_data = ps_data.to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            z = torch.randn((sample_size, 3)).to(self.device)
            cond = ps_data.repeat(sample_size, 1)
            samples, _ = self.flow_model(z, c=[cond], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            #samples = torch.clamp(samples, 0, 1)
            samples_np = samples.detach().cpu().numpy()
     
            true_np = true_label.detach().cpu().numpy().flatten()
            for j in range(3):
                # Count how many posterior samples are lower than the true value.
                rank_val = np.sum(samples_np[:, j] < true_np[j])
                ranks[param_names[j]].append(rank_val / sample_size)
            if i % 100 == 0:
                logging.info(f"Processed {i} samples for individual rank plots.")
        with PdfPages(os.path.join(self.output_dir, 'individual_ranks.pdf')) as pdf:
            for idx, param in enumerate(param_names):
                plt.figure(figsize=(8, 6))
                plt.hist(ranks[param], bins=bins, density=True, alpha=0.7,
                         color='green', edgecolor='black')
                plt.xlabel('Normalized Rank')
                plt.ylabel('Density')
                plt.title(f'Rank Histogram for {param}')
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        logging.info("Individual rank plots saved.")

        def plot_predicted_true_histograms(self, sample_size: int = 1000):
            """
            For each test sample, use the stored CNN condition to sample from the flow,
            aggregate all predicted xH values (without averaging), and then for each xH parameter
            create a plot that shows (i) the histogram of all flow-generated samples and (ii)
            the histogram of the true labels (across the test set). Also log the number of predictions
            below 0 and above 1.
            """
            # Ensure true labels are loaded.
            self.get_data()  # This sets self.labels (shape: [num_samples, 3])
            num_samples = len(self.dataset)
            all_predicted = []  # will collect predictions per sample; shape: (num_samples*sample_size, n_labels)
            
            # Loop over test samples.
            for i, (ps_data, _) in enumerate(self.data_loader):
                ps_data = ps_data.to(self.device)
                # Create a condition by repeating the test sample.
                cond = ps_data.repeat(sample_size, 1)
                # Sample from the base distribution.
                z = torch.randn((sample_size, self.n_labels)).to(self.device) * np.sqrt(self.temperature)
                # Generate flow predictions in reverse mode.
                samples, _ = self.flow_model(z, c=[cond], rev=True)
                if self.sigmoid:
                    epsilon = 1e-5
                    # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                    xH_adj = torch.sigmoid(samples)
                    # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                    samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            #samples = torch.clamp(samples, 0, 1)
                # Clamp predictions to [0, 1] for diagnostic purposes.
                #samples = torch.clamp(samples, 0, 1)
                # Convert to numpy array.
                samples_np = samples.detach().cpu().numpy()
                all_predicted.append(samples_np)
                if i % 100 == 0:
                    logging.info(f"Aggregated predictions for sample {i}.")
            
            # Concatenate predictions across all test samples.
            all_predicted = np.concatenate(all_predicted, axis=0)  # shape: (num_samples*sample_size, n_labels)
            true_labels = self.labels  # shape: (num_samples, n_labels)
            
            # For each parameter, plot the histogram of predicted samples and true labels.
            param_names = ['xH1', 'xH2', 'xH3']
            for j in range(self.n_labels):
                plt.figure(figsize=(8,6))
                # Histogram of flow-predicted samples.
                plt.hist(all_predicted[:, j], bins=100, alpha=0.7, density=True, label='Flow Predictions', color='blue')
                # Histogram of true labels.
                plt.hist(true_labels[:, j], bins=100, alpha=0.7, density=True, label='True Labels', color='orange')
                plt.xlabel(param_names[j])
                plt.ylabel('Density')
                plt.title(f'Histogram of {param_names[j]}: Flow Samples vs. True Labels')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(self.output_dir, f'{param_names[j]}_predicted_true_histogram.pdf')
                plt.savefig(plot_path)
                plt.close()
                # Count out-of-bound predictions 
                num_below = np.sum(all_predicted[:, j] < 0)
                num_above = np.sum(all_predicted[:, j] > 1)
                logging.info(f"For {param_names[j]}: {num_below} values below 0, {num_above} values above 1 out of {all_predicted.shape[0]} samples.")


    def main(self) -> None:
        """
        Run the full inference pipeline:
          1. Calculate posterior statistics.
          2. Generate the calibration scatter plot.
          3. Generate the log-probability coverage line plot.
          4. Plot the individual parameter rank histograms.
          5. Produce corner plots for example test samples.
        """
        logging.info("Starting inference and plotting...")
        self.calc_statistics()
        self.plot_calibration_scatter()
        self.plot_log_prob_coverage_line()
        self.plot_individual_ranks()
        self.plot_corner(sample_indices=list(range(10)))
        #self.plot_predicted_true_histograms(sample_size=1000)
        logging.info("Inference and plotting complete.")


if __name__ == '__main__':
    # Set up logging to the console.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define the model directory and parameters.
    model_dir = output_dir
    params = {
        'plot': {
            'plot_dir': os.path.join(model_dir, 'plots_ast')
        }
    }

    model_params = {
        'flow': {
            'n_dim': 3,
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': 303,  # Adjust based on the flattened 2DPS input size.
            'dropout': 0.0,
            'load': False,
            'model_location': 'trained_model.pth',
            'sigmoid': False
        }
    }
    
    # Load the trained flow model.
    flow_model = ConditionalInvertibleBlock(model_params)
    trained_model_path = os.path.join(model_dir, 'best_model.pth')
    flow_model.flow.load_state_dict(torch.load(trained_model_path))
    flow_model.flow.eval()
    
    # Specify the test data directory (can be a list of folders).
    #test_data_dir = ['/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/data_test']
    test_data_dir = ['/lustre/fswork/projects/rech/ybg/uuv28wh/astraeus_data/power_spectra_stitched']
    
    # Create an instance of the InferenceModel.
    inference_model = InferenceModel(params=params,
                                     flow_model=flow_model,
                                     data_dir=test_data_dir,
                                     device='cuda' if torch.cuda.is_available() else 'cpu',
                                     sigmoid=True,
                                     temperature=1)
    
    # Run inference and generate all plots.
    inference_model.main()