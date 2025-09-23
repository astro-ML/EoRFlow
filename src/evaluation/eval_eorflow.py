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
from data_loader import PowerSpectrumDataset, EoRFlowDataset
from matplotlib.backends.backend_pdf import PdfPages
from getdist import plots, MCSamples

# ------------------- CONFIG -------------------
mode = 'ps2d'  # Options: 'ps2d' or 'ps1d'
n_blocks = 10
n_nodes = 512

out_tag = f'{n_blocks}_{n_nodes}'
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/aa4_mod_10_512'

# set min and max redshift index 
min_redshift_index = 0
max_redshift_index = 15
redshift_dim = max_redshift_index - min_redshift_index

if mode == 'ps2d':
    ps_dim = redshift_dim * 10 * 10
else:
    ps_dim = redshift_dim * 10
cond_dims = ps_dim + redshift_dim

# ------------------- SETUP -------------------
output_plot_dir = os.path.join(model_dir, 'plots')
os.makedirs(output_plot_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- Inference Model -------------------
class InferenceModel:
    def __init__(self, params, flow_model, data_dirs, device='cpu', use_sigmoid=False):
        self.params = params.get('plot', {})
        self.flow_model = flow_model.flow.to(device)
        self.device = device
        self.use_sigmoid = use_sigmoid

        # redshift dims
        self.min_redshift_index = params['min_redshift_index']
        self.max_redshift_index = params['max_redshift_index']
        self.redshift_dim = self.max_redshift_index - self.min_redshift_index

        self.redshift_start = self.params.get('redshift_start', 5)
        self.redshift_end = self.params.get('redshift_end', 12)

        # DATASET
        self.dataset = PowerSpectrumDataset(
            data_dirs=data_dirs,
            mode=mode,
            logit=False,
            add_noise=False,
            aa4_mod_noise=True,
            aaStar_mod_noise=False,
            min_redshift_index=self.min_redshift_index,
            max_redshift_index=self.max_redshift_index
        )
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # OUTPUT
        self.output_dir = self.params.get('plot_dir', output_plot_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.labels = None
        self.all_samples = []

    def get_data(self) -> np.ndarray:
        """
        Loads the true labels from the dataset.

        Returns:
            np.ndarray: Array of true labels with shape (num_samples, redshift_dim).
        """
        num_samples = len(self.dataset)
        self.labels = np.zeros((num_samples, self.redshift_dim))
        for i, (_, label) in enumerate(self.data_loader):
            self.labels[i] = label.detach().cpu().numpy().flatten()
        return self.labels

    def precompute_samples(self, sample_size: int = 1000) -> None:
        """
        Compute and cache the posterior samples for every test sample.
        This is done once so that all subsequent evaluations use these samples.

        Args:
            sample_size (int): Number of posterior samples drawn per test sample.
        """
        self.all_samples = []
        self.get_data()  # Ensure self.labels is set
        logging.info("Precomputing posterior samples for all test samples...")
        with torch.no_grad():
            for idx, (ps_data, true_label) in enumerate(self.data_loader):
                ps_data = ps_data.to(self.device)  # shape: (1, cond_dim)
                true_label = true_label.to(self.device)  # shape: (1, redshift_dim)
                # Repeat condition for sampling.
                cond = ps_data  # use ps_data as condition
                cond_repeated = cond.repeat(sample_size, 1)
                z = torch.randn((sample_size, self.redshift_dim)).to(self.device)
                samples, _ = self.flow_model(z, c=[cond_repeated], rev=True)
                if self.use_sigmoid:
                    epsilon = 1e-5
                    samples = torch.sigmoid(samples)
                    samples = (samples - epsilon) / (1 - 2 * epsilon)
                samples_np = samples.detach().cpu().numpy()  # (sample_size, redshift_dim)
                true_np = true_label.detach().cpu().numpy().flatten()  # (redshift_dim,)
                # Store along with the original condition (ps_data)
                self.all_samples.append({
                    'samples': samples_np,
                    'true_label': true_np,
                    'ps_data': ps_data.detach().cpu().numpy()  # may be used for log-prob computation
                })
                if idx % 100 == 0:
                    logging.info(f"Precomputed samples for {idx} test samples.")
        logging.info("Posterior samples precomputation complete.")

    def calc_statistics(self, output_filename: str = 'inference_statistics_xH.npz') -> None:
        """
        Calculate statistics (mean, 16th and 84th percentiles) for each test sample.
        Uses the precomputed samples.
        Saves the results (with the true labels) to a file.
        """
        num_samples = len(self.all_samples)
        mean = np.zeros((num_samples, self.redshift_dim))
        lower = np.zeros((num_samples, self.redshift_dim))
        upper = np.zeros((num_samples, self.redshift_dim))
        par_names = [f"xHI{i+1}" for i in range(self.redshift_dim)]
        for i, sample_dict in enumerate(self.all_samples):
            samples = sample_dict['samples']  # shape: (sample_size, redshift_dim)
            mc_samples = MCSamples(samples=samples, names=par_names, labels=par_names,
                                    settings={"fine_bins_2D": 300, "fine_bins_1D": 300,
                                    "smooth_scale_2D": 0.1, "smooth_scale_1D": 0.1})
            stats = mc_samples.getMargeStats()
            for j, name in enumerate(par_names):
                par_stat = stats.parWithName(name)
                if par_stat is None:
                    logging.warning(f"Parameter {name} not found in the statistics!")
                else:
                    mean[i, j] = par_stat.mean
                    lower[i, j] = par_stat.limits[0].lower
                    upper[i, j] = par_stat.limits[0].upper
            if i % 100 == 0:
                logging.info(f"Computed statistics for {i} samples.")
                
        np.savez(os.path.join(self.output_dir, output_filename),
                 mean=mean, lower=lower, upper=upper, label=self.labels)
        logging.info("Inference statistics saved.")

    def plot_calibration_scatter(self, stats_filename: str = 'inference_statistics_xH.npz') -> None:
        """
        Plot a calibration scatter plot comparing true vs. inferred values.
        """
        data = np.load(os.path.join(self.output_dir, stats_filename))
        labels = data['label']
        mean = data['mean']
        lower = data['lower']
        upper = data['upper']
        error_low = np.abs(mean - lower)
        error_up = np.abs(upper - mean)
        par_names = [f"xHI{i+1}" for i in range(self.redshift_dim)]
        with PdfPages(os.path.join(self.output_dir, 'calibration_scatter.pdf')) as pdf:
            for idx, param in enumerate(par_names):
                plt.figure(figsize=(8, 6))
                plt.errorbar(labels[:, idx], mean[:, idx],
                             yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='o', color='blue', ecolor='lightblue', capsize=3,
                             label='Inferred')
                plt.plot(labels[:, idx], labels[:, idx], 'r--', label='Ideal')
                plt.xlabel('True Value')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.ylabel(f'Inferred {param}')
                plt.title(f'Calibration Scatter Plot for {param}')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
        logging.info("Calibration scatter plot saved.")

    def plot_corner(self, sample_indices: list = None) -> None:
        """
        Generate corner plots (using getdist) for selected test samples.
        """
        if sample_indices is None:
            sample_indices = list(range(5))
        par_names = [f"xHI{i+1}" for i in range(self.redshift_dim)]
        for idx in sample_indices:
            sample_dict = self.all_samples[idx]
            samples = sample_dict['samples']
            true_label = sample_dict['true_label']
            mc_samples = MCSamples(samples=samples, names=par_names, labels=par_names,
                                    settings={"fine_bins_2D": 300, "fine_bins_1D": 300,
                                            "smooth_scale_2D": 0.1, "smooth_scale_1D": 0.1})
            g = plots.get_subplot_plotter()
            g.triangle_plot([mc_samples], filled=True)
            for i, name in enumerate(par_names):
                g.subplots[i, i].axvline(true_label[i], color='red', linestyle='--')
            for i in range(self.redshift_dim - 1):
                for j in range(i + 1, self.redshift_dim):
                    ax = g.subplots[j, i]
                    ax.plot(true_label[i], true_label[j], 'ro', markersize=6)
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

    def plot_log_prob_coverage_line(self) -> None:
        """
        Compute and plot a log-probability coverage line.
        Uses the precomputed samples.
        """
        ranks = []
        for i, sample_dict in enumerate(self.all_samples):
            ps_data = sample_dict['ps_data']  # shape: (1, cond_dim)
            true_label = torch.tensor(sample_dict['true_label']).unsqueeze(0).to(self.device)
            # Compute true log-probability.
            true_log_prob = self.compute_log_prob(true_label, torch.tensor(ps_data).to(self.device))
            # Compute log-probs for samples.
            samples_tensor = torch.tensor(sample_dict['samples']).to(self.device)
            # For condition, repeat ps_data to match sample size.
            cond_repeated = torch.tensor(ps_data).to(self.device).repeat(samples_tensor.shape[0], 1)
            sample_log_probs = self.compute_log_prob(samples_tensor, cond_repeated)
            rank = (sample_log_probs < true_log_prob).float().mean().item()
            ranks.append(rank)
            if i % 100 == 0:
                logging.info(f"Computed log-prob coverage for {i} samples.")
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


    def credible_interval(self, x: np.ndarray, alpha: float) -> tuple[float, float]:
        """
        Calculate the credible interval for a given set of samples for one parameter.

        Parameters:
            x (np.ndarray): The samples for which the credible interval is calculated.
            alpha (float): The desired credible level (between 0 and 1).
            index (int): The index of the parameter in the density grid.

        Returns:
            Tuple[float, float]: The lower and upper bounds of the credible interval.
        """
        # Reshape to 2D array with one column for getdist
        x_mc = MCSamples(samples=x.reshape(-1, 1), settings={"fine_bins_2D": 300, "fine_bins_1D": 300,
                                    "smooth_scale_2D": 0.1, "smooth_scale_1D": 0.1})
        grid = x_mc.get1DDensityGridData(0)
        low, up = grid.getLimits([alpha])[0:2]
        return low, up
        
    def plot_individual_ranks(self, bins: int = 20) -> None:
        """
        Plot histograms of the individual parameter ranks over the test set.
        Uses the precomputed samples.
        """
        ranks = {f"xHI{i+1}": [] for i in range(self.redshift_dim)}
        par_names = [f"xHI{i+1}" for i in range(self.redshift_dim)]
        for i, sample_dict in enumerate(self.all_samples):
            samples = sample_dict['samples']
            true_np = sample_dict['true_label']
            for j in range(self.redshift_dim):
                rank_val = np.sum(samples[:, j] < true_np[j])
                ranks[par_names[j]].append(rank_val / samples.shape[0])
            if i % 100 == 0:
                logging.info(f"Computed individual ranks for {i} samples.")
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

    def plot_reionization_history(self) -> None:
        """
        Generate reionization history plots (xHI vs. redshift) for selected examples.
        Uses the precomputed samples.
        """
        redshift_values = np.linspace(self.redshift_start, self.redshift_end, self.redshift_dim)
        n_examples_to_plot = min(5, len(self.all_samples))
        for i in range(n_examples_to_plot):
            sample_dict = self.all_samples[i]
            samples = sample_dict['samples']
            true_xH = sample_dict['true_label']
            mean_prediction = np.mean(samples, axis=0)
            std_prediction = np.std(samples, axis=0)
            plt.figure(figsize=(12, 8))
            plt.plot(redshift_values, mean_prediction, 'b-', linewidth=2, label='Mean Prediction')
            plt.fill_between(redshift_values,
                             mean_prediction - std_prediction,
                             mean_prediction + std_prediction,
                             color='b', alpha=0.3, label='1$\sigma$ Interval')
            plt.plot(redshift_values, true_xH, 'ro-', linewidth=2, label='True xHI')
            plt.xlabel('Redshift (z)', fontsize=14)
            plt.ylabel('Neutral Hydrogen Fraction (xHI)', fontsize=14)
            plt.title(f'Reionization History for Example {i}', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            out_path = os.path.join(self.output_dir, f'reionization_history_example_{i}.pdf')
            plt.savefig(out_path)
            plt.close()
            logging.info(f"Reionization history plot saved for example {i} at {out_path}")

    def main(self, sample_size: int = 1000) -> None:
        """
        Run the full inference pipeline:
          1. Precompute posterior samples.
          2. Calculate posterior statistics.
          3. Generate calibration scatter plot.
          4. Generate log-probability coverage line plot.
          5. Plot individual parameter rank histograms.
          6. Produce corner plots.
          7. Plot reionization history.
        """
        logging.info("Starting inference and plotting...")
        self.precompute_samples(sample_size=sample_size)
        self.calc_statistics()
        self.plot_calibration_scatter()
        self.plot_log_prob_coverage_line()
        #self.plot_marginal_log_prob_coverage()
        self.plot_individual_ranks()
        self.plot_corner()
        self.plot_reionization_history()
        logging.info("Inference and plotting complete.")


# ------------------- MAIN -------------------
if __name__ == '__main__':
    # Build params dict
    params = {
        'plot': {
            'plot_dir': output_plot_dir
        },
        'min_redshift_index': min_redshift_index,
        'max_redshift_index': max_redshift_index,
        'redshift_start': 5.0,
        'redshift_end': 12.0,
    }

    model_params = {
        'flow': {
            'n_dim': redshift_dim,
            'n_blocks': n_blocks,
            'n_nodes': n_nodes,
            'cond_dims': cond_dims,
            'load': False,
            'model_location': 'best_flow_model.pth',
        }
    }

    # Load model
    flow_model = ConditionalInvertibleBlock(model_params)
    flow_model.flow.load_state_dict(
        torch.load(os.path.join(model_dir, 'best_flow_model.pth'), map_location=device)
    )
    flow_model.flow.eval()

    # Test data
    test_data_dir = [f'/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/pure/test']

    # Inference
    inference_model = InferenceModel(
        params=params,
        flow_model=flow_model,
        data_dirs=test_data_dir,
        device=device,
        use_sigmoid=True
    )
    inference_model.main(sample_size=1000)