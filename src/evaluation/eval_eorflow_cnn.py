import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import numpy as np
import logging
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset
from matplotlib.backends.backend_pdf import PdfPages
from getdist import plots, MCSamples

# Import your CNN model (adjust as needed)
from cnn import CNN3D_15 as CNN

class InferenceModelCNN:
    """
    InferenceModelCNN class for performing inference with a trained CNN+Flow model.

    This class produces:
      - Calibration scatter plots: true vs. inferred xH values (with error bars).
      - Corner plots: 2D posterior contours for selected test samples.
      - Individual rank histograms: distribution of the rank of the true value among posterior samples.
      - Log-probability coverage line plot: comparing the empirical CDF of (1 - rank) against the ideal uniform CDF.
      - Reionization history plots: xH as a function of redshift for selected examples.

    Args:
        params (dict): Dictionary with plotting parameters and redshift settings.
            Expected keys include:
              - 'plot': dict containing 'plot_dir'
              - 'min_redshift_index': int (default: 4)
              - 'max_redshift_index': int (default: 15)
              - Optionally, 'redshift_start' and 'redshift_end' (for plotting xH vs. z).
        flow_model (nn.Module): Trained flow model. It is expected to have a .flow attribute.
        cnn_model (nn.Module): Trained CNN summary network.
        data_dir (str or list): Path(s) to the test dataset.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    def __init__(self, params: dict, flow_model: torch.nn.Module, cnn_model: torch.nn.Module, data_dir, 
                device: str = 'cpu', sigmoid=False):
        self.params = params.get('plot', {})
        self.flow_model = flow_model.flow.to(device)
        self.cnn_model = cnn_model.to(device)
        self.device = device
        self.sigmoid = sigmoid
        # Set redshift parameters (min and max indices) and compute redshift_dim.
        self.min_redshift_index = params.get('min_redshift_index', 4)
        self.max_redshift_index = params.get('max_redshift_index', 15)
        self.redshift_dim = self.max_redshift_index - self.min_redshift_index

        self.redshift_start = self.params.get('redshift_start', 6.25)
        self.redshift_end = self.params.get('redshift_end', 8.41)

        # Initialize dataset and data loader.
        # In CNN+Flow mode, the dataset is expected to return (ps_data, label, redshift_batch).
        self.dataset = PowerSpectrumDataset(
            data_dir,
            max_ones_allowed=15,
            max_zeros_allowed=15,
            filter_reionization_timing=False,
            min_redshift_index=self.min_redshift_index,
            max_redshift_index=self.max_redshift_index,
            use_cnn=True
        )
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Setup output directory.
        self.output_dir = self.params.get('plot_dir', './plots')
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for true labels.
        self.labels = None

    def get_data(self) -> np.ndarray:
        """
        Loads the true labels from the dataset.

        Returns:
            np.ndarray: Array of true labels with shape (num_samples, redshift_dim).
        """
        num_samples = len(self.dataset)
        self.labels = np.zeros((num_samples, self.redshift_dim))
        for i, (_, label, _) in enumerate(self.data_loader):
            self.labels[i] = label.detach().cpu().numpy().flatten()
        return self.labels

    def _get_condition(self, batch, sample_size: int) -> torch.Tensor:
        """
        Construct the conditioning variable using the CNN summary network.
        
        Expects batch to contain (ps_data, label, redshift_batch). The ps_data tensor is unsqueezed 
        (to add a channel dimension) and passed through the CNN together with the redshift_batch.
        The output is then concatenated with redshift_batch to form the condition.
        The condition is repeated for the number of samples to be generated.
        """
        ps_data, _, redshift_batch = batch
        ps_data = ps_data.to(self.device)
        redshift_batch = redshift_batch.to(self.device)
        if ps_data.ndim == 3:  # (15, 10, 10) → need to add batch and channel
            ps_data_cnn = ps_data.unsqueeze(0).unsqueeze(0)  # → (1, 1, 15, 10, 10)
        elif ps_data.ndim == 4:  # already batched: (B, 15, 10, 10)
            ps_data_cnn = ps_data.unsqueeze(1)  # → (B, 1, 15, 10, 10)
        else:
            raise ValueError(f"Unexpected ps_data shape: {ps_data.shape}")
        if redshift_batch.ndim == 1:
            redshift_batch = redshift_batch.unsqueeze(0)
        cnn_output = self.cnn_model(ps_data_cnn, redshift_batch)
        condition = torch.cat([cnn_output, redshift_batch], dim=1)
        return condition.repeat(sample_size, 1)

    def calc_statistics(self, output_filename: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """
        Calculate Monte Carlo statistics (mean, 16th and 84th percentiles) for the posterior of each sample.
        Saves the results (with the true labels) to a file.
        """
        self.get_data()
        num_samples = len(self.dataset)
        mean = np.zeros((num_samples, self.redshift_dim))
        lower = np.zeros((num_samples, self.redshift_dim))
        upper = np.zeros((num_samples, self.redshift_dim))
        par_names = [f"xH{i+1}" for i in range(self.redshift_dim)]
        
        for i, batch in enumerate(self.data_loader):
            condition = self._get_condition(batch, sample_size)
            z = torch.randn((sample_size, self.redshift_dim)).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            if self.sigmoid:
                logging.info('Using Sigmoid transformation...')
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            samples = samples.detach().cpu().numpy()
            for j in range(self.redshift_dim):
                if samples[:, j].max() - samples[:, j].min() <= 0:
                    samples[:, j] += np.random.normal(0, 1e-6, sample_size)
            mc_samples = MCSamples(samples=samples, names=par_names, labels=par_names)
            stats = mc_samples.getMargeStats()
            for j, name in enumerate(par_names):
                par_stat = stats.parWithName(name)
                if par_stat is None:
                    print(f"Warning: Parameter {name} not found!")
                else:
                    mean[i, j] = par_stat.mean
                    lower[i, j] = par_stat.limits[0].lower
                    upper[i, j] = par_stat.limits[0].upper
            if i % 100 == 0:
                logging.info(f"Processed {i} samples for statistics.")
        
        np.savez(os.path.join(self.output_dir, output_filename),
                 mean=mean, lower=lower, upper=upper, label=self.labels)
        logging.info("Inference statistics saved.")
        
    def plot_calibration_scatter(self, stats_filename: str = 'inference_statistics_xH.npz') -> None:
        """
        Plot a calibration scatter plot comparing true vs. inferred xH values.
        """
        data = np.load(os.path.join(self.output_dir, stats_filename))
        labels = data['label']
        mean = data['mean']
        lower = data['lower']
        upper = data['upper']
        error_low = np.abs(mean - lower)
        error_up = np.abs(upper - mean)
        par_names = [f"xH{i+1}" for i in range(self.redshift_dim)]
        with PdfPages(os.path.join(self.output_dir, 'calibration_scatter.pdf')) as pdf:
            for idx, param in enumerate(par_names):
                plt.figure(figsize=(8, 6))
                plt.errorbar(labels[:, idx], mean[:, idx],
                             yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='o', color='blue', ecolor='lightblue', capsize=3,
                             label='Inferred')
                plt.plot(labels[:, idx], labels[:, idx], 'r--', label='Ideal')
                plt.xlabel('True Value')
                plt.ylabel(f'Inferred {param}')
                plt.title(f'Calibration Scatter for {param}')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
        logging.info("Calibration scatter plot saved.")
        
    def plot_corner(self, sample_indices: list = None, sample_size: int = 1000) -> None:
        """
        Generate corner plots for selected test samples.
        """
        if sample_indices is None:
            sample_indices = list(range(5))
        par_names = [f"xH{i+1}" for i in range(self.redshift_dim)]
        for idx in sample_indices:
            ps_data, true_label, redshift_batch = self.dataset[idx]
            ps_data = ps_data.to(self.device)
            redshift_batch = redshift_batch.to(self.device)
            batch = (ps_data, true_label, redshift_batch)
            condition = self._get_condition(batch, sample_size)
            z = torch.randn((sample_size, self.redshift_dim)).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            samples = samples.detach().cpu().numpy()
            mc_samples = MCSamples(samples=samples, names=par_names, labels=par_names)
            g = plots.get_subplot_plotter()
            g.triangle_plot([mc_samples], filled=True)
            true_values = true_label.detach().cpu().numpy().flatten()
            for i, name in enumerate(par_names):
                g.subplots[i, i].axvline(true_values[i], color='red', linestyle='--')
            for i in range(self.redshift_dim - 1):
                for j in range(i + 1, self.redshift_dim):
                    ax = g.subplots[j, i]
                    ax.plot(true_values[i], true_values[j], 'ro', markersize=6)
            plot_path = os.path.join(self.output_dir, f'corner_plot_sample_{idx+1}.pdf')
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Corner plot saved for sample {idx+1}.")
        
    def compute_log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-probability of x under the flow model given condition c.
        """
        if c.dim() == 1:
            c = c.unsqueeze(0)
        z, log_det = self.flow_model(x, c=[c], rev=False)
        base_log_prob = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1)
        return base_log_prob + log_det

    def plot_log_prob_coverage_line(self, sample_size: int = 1000) -> None:
        """
        Compute and plot a log-probability coverage line.
        """
        ranks = []
        for i, batch in enumerate(self.data_loader):
            ps_data, true_label, redshift_batch = batch
            ps_data = ps_data.to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            condition = self._get_condition(batch, sample_size)
            # Pass the full repeated condition (not condition[0]) so that its batch size matches the samples.
            true_log_prob = self.compute_log_prob(true_label, condition[0:1] )
            z = torch.randn((sample_size, self.redshift_dim)).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            sample_log_probs = self.compute_log_prob(samples, condition)
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
        Plot histograms of individual parameter ranks over the test set.
        """
        ranks = {f"xH{i+1}": [] for i in range(self.redshift_dim)}
        par_names = [f"xH{i+1}" for i in range(self.redshift_dim)]
        for i, batch in enumerate(self.data_loader):
            ps_data, true_label, redshift_batch = batch
            ps_data = ps_data.to(self.device)
            true_label = true_label.to(self.device).view(1, -1)
            condition = self._get_condition(batch, sample_size)
            z = torch.randn((sample_size, self.redshift_dim)).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            samples_np = samples.detach().cpu().numpy()
            true_np = true_label.detach().cpu().numpy().flatten()
            for j in range(self.redshift_dim):
                rank_val = np.sum(samples_np[:, j] < true_np[j])
                ranks[par_names[j]].append(rank_val / sample_size)
            if i % 100 == 0:
                logging.info(f"Processed {i} samples for individual rank plots.")
        with PdfPages(os.path.join(self.output_dir, 'individual_ranks.pdf')) as pdf:
            for param in par_names:
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

    def plot_reionization_history(self, sample_size: int = 1000) -> None:
        """
        Generate reionization history plots (xH vs. redshift) for a number of examples.
        """
        redshift_values = np.linspace(self.redshift_start, self.redshift_end, self.redshift_dim)
        n_total = len(self.dataset)
        n_examples = min(5, n_total)
        for i in range(n_examples):
            ps_data, true_label, redshift_batch = self.dataset[i]
            ps_data = ps_data.to(self.device)
            redshift_batch = redshift_batch.to(self.device)
            batch = (ps_data, true_label, redshift_batch)
            condition = self._get_condition(batch, sample_size)
            z = torch.randn((sample_size, self.redshift_dim)).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            if self.sigmoid:
                epsilon = 1e-5
                # Apply the sigmoid (inverse of the logit) to get values in [epsilon, 1-epsilon]
                xH_adj = torch.sigmoid(samples)
                # Undo the squeezing: convert xH_adj back to xH in [0, 1]
                samples = (xH_adj - epsilon) / (1 - 2 * epsilon)
            #samples = torch.clamp(samples, 0, 1)
            samples = samples.detach().cpu().numpy()
            true_xH = true_label.detach().cpu().numpy().flatten()
            mean_prediction = np.mean(samples, axis=0)
            std_prediction = np.std(samples, axis=0)
            
            plt.figure(figsize=(12, 8))
            plt.plot(redshift_values, mean_prediction, 'b-', linewidth=2, label='Mean Prediction')
            plt.fill_between(redshift_values,
                             mean_prediction - std_prediction,
                             mean_prediction + std_prediction,
                             color='b', alpha=0.3, label='1$\sigma$ Interval')
            plt.plot(redshift_values, true_xH, 'ro-', linewidth=2, label='True xH')
            plt.xlabel('Redshift (z)', fontsize=14)
            plt.ylabel('Neutral Hydrogen Fraction (xH)', fontsize=14)
            plt.title(f'Reionization History for Example {i}', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            out_path = os.path.join(self.output_dir, f'reionization_history_example_{i}.pdf')
            plt.savefig(out_path)
            plt.close()
            logging.info(f"Reionization history plot saved for example {i} at {out_path}")

    def main(self) -> None:
        """
        Run the full CNN+Flow inference pipeline:
          1. Calculate posterior statistics.
          2. Generate calibration scatter plot.
          3. Generate log-probability coverage line plot.
          4. Plot individual parameter rank histograms.
          5. Produce corner plots.
          6. Plot reionization history.
        """
        logging.info("Starting CNN+Flow inference and plotting...")
        self.calc_statistics()
        self.plot_calibration_scatter()
        self.plot_log_prob_coverage_line()
        self.plot_individual_ranks()
        self.plot_corner()
        self.plot_reionization_history()
        logging.info("CNN+Flow inference and plotting complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set redshift indices via parameters.
    min_redshift_index = 0
    max_redshift_index = 15
    redshift_dim = max_redshift_index - min_redshift_index

    ps_dim = redshift_dim * 10 * 10  # Flattened power spectra dimension.
    total_cond_dim = 9 + redshift_dim  # For CNN+Flow, adjust if needed.

    model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_cnn_logit/pure_8_256_-5_9cnn'
    params = {
        'plot': {
            'plot_dir': os.path.join(model_dir, 'plots')
        },
        'min_redshift_index': min_redshift_index,
        'max_redshift_index': max_redshift_index,
        'redshift_start': 8.0,
        'redshift_end': 10.0,
        'dims': {
            'n_dim': redshift_dim
        }
    }

    model_params = {
        'flow': {
            'n_dim': redshift_dim,
            'n_blocks': 8,
            'n_nodes': 256,
            # For CNN+Flow, cond_dims should match the output dimension of your CNN plus the redshift dimension.
            'cond_dims': total_cond_dim,  
            'dropout': 0.0,
            'load': False,
            'model_location': 'best_flow_model.pth',
            'sigmoid': False
        }
    }

    # Load the trained flow model.
    flow_model = ConditionalInvertibleBlock(model_params)
    trained_flow_path = os.path.join(model_dir, 'best_flow_model.pth')
    flow_model.flow.load_state_dict(torch.load(trained_flow_path))
    flow_model.flow.eval()

    # Load the trained CNN model.
    cnn_model = CNN()
    cnn_weights_path = os.path.join(model_dir, 'best_cnn_model.pth')
    cnn_model.load_state_dict(torch.load(cnn_weights_path))
    cnn_model.eval()

    #test_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/test']
    test_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test']

    inference_model = InferenceModelCNN(params=params,
                                        flow_model=flow_model,
                                        cnn_model=cnn_model,
                                        data_dir=test_data_dir,
                                        device='cuda' if torch.cuda.is_available() else 'cpu',
                                        sigmoid = True)
    
    inference_model.main()