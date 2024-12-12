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

    Args:
        params (dict): Dictionary containing model parameters.
        flow_model (nn.Module): The trained flow model.
        data_dir (str): Directory for the test data.
        device (str, optional): Device for computation ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(self, params: dict, flow_model: nn.Module, data_dir: str, device: str = 'cpu') -> None:
        self.params = params['plot']
        self.flow_model = flow_model.flow.to(device)
        self.device = device

        # Initialize dataset and data loader
        self.dataset = PowerSpectrumDataset(data_dir)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Set up output directory
        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for labels
        self.label = None

    def get_data(self) -> np.ndarray:
        """Load test data and labels."""
        num_samples = len(self.dataset)
        self.label = np.zeros((num_samples, 3))  # Assuming you are inferring 3 xH values

        for i, (ps_data, true_label) in enumerate(self.data_loader):
            self.label[i] = true_label.detach().cpu().numpy()

        return self.label

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """Calculate test statistics for inferred xH values."""
        self.label = self.get_data()
        logging.info('Calculating statistics for xH inference.')

        mean, lower, upper = np.zeros_like(self.label), np.zeros_like(self.label), np.zeros_like(self.label)

        for i, ps_data in enumerate(self.data_loader):
            ps_data = ps_data[0].to(self.device)  # Extract 2DPS data from the batch

            z = torch.randn((sample_size, 3)).to(self.device)
            samples, _ = self.flow_model(z, c=[ps_data.repeat((sample_size, 1)).to(self.device)], rev=True)
            samples = samples.detach().cpu().numpy()

            mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])
            stats = mc_samples.getMargeStats()

            for j, name in enumerate(['xH1', 'xH2', 'xH3']):
                mean[i, j] = stats.parWithName(name).mean
                lower[i, j] = stats.parWithName(name).limits[0].lower
                upper[i, j] = stats.parWithName(name).limits[0].upper

            if i % 100 == 0:
                logging.info(f"Processed {i} samples.")

        # Save statistics
        np.savez(os.path.join(self.output_dir, output_name), mean=mean, lower=lower, upper=upper, label=self.label)
        logging.info('Statistics saved.')

    def plot_calibration(self) -> None:
        """Plot calibration curve comparing true vs. inferred xH values."""
        data = np.load(os.path.join(self.output_dir, 'inference_statistics_xH.npz'))
        label, mean, lower, upper = data['label'], data['mean'], data['lower'], data['upper']
        error_low, error_up = np.abs(mean - lower), np.abs(upper - mean)

        with PdfPages(os.path.join(self.output_dir, 'calibration_xH.pdf')) as pdf:
            for idx, param in enumerate(['xH1', 'xH2', 'xH3']):
                plt.figure(figsize=(8, 6))
                plt.errorbar(label[:, idx], mean[:, idx], yerr=[error_low[:, idx], error_up[:, idx]],
                             fmt='.', color='blue', ecolor='lightblue', elinewidth=1, capsize=2)
                plt.plot(label[:, idx], label[:, idx], color='red', linestyle='--')
                plt.xlabel('True Value')
                plt.ylabel(f'Inferred {param}')
                plt.title(f'Calibration plot for {param}')
                pdf.savefig()
                plt.close()
        logging.info("Calibration plot saved.")

    def plot_corner(self, index: int, sample_size: int = 1000) -> None:
        """Generate corner plot for a single test sample's xH posteriors."""
        z = torch.randn((sample_size, 3)).to(self.device)
        ps_data, true_label = self.dataset[index]
        ps_data = ps_data.to(self.device)
        
        samples, _ = self.flow_model(z, c=[ps_data.repeat((sample_size, 1)).to(self.device)], rev=True)
        samples = samples.detach().cpu().numpy()
        mc_samples = MCSamples(samples=samples, names=['xH1', 'xH2', 'xH3'])

        g = plots.get_subplot_plotter()
        g.triangle_plot([mc_samples], filled=True)

        true_values = true_label.numpy()
        for i, name in enumerate(['xH1', 'xH2', 'xH3']):
            g.subplots[i, i].axvline(true_values[i], color='red', linestyle='--')

        # Add true values as dots in the 2D contour plots
        for i in range(2):  # Loop through parameters for the x-axis
            for j in range(i + 1, 3):  # Loop through parameters for the y-axis
                ax = g.subplots[j, i]
                ax.plot(true_values[i], true_values[j], 'ro', markersize=6)  # Add red dot at the true value

        plot_path = os.path.join(self.output_dir, f'corner_plot_xH_sample_{index + 1}.pdf')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Corner plot saved for sample {index + 1}.")

    def main(self) -> None:
        """Main function to run inference and create plots."""
        self.calc_statistics(output_name='inference_statistics_xH.npz')
        self.plot_calibration()
        for i in range(5):  # Generate corner plots for the first 5 samples as an example
            self.plot_corner(index=i)
        logging.info("Inference and plotting complete.")


# Define the model directory and paths
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/flow_redshift_8_blocks'

# Parameters dictionary setup
params = {
    'plot': {
        'data_path': '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10',
        'plot_dir': model_dir + '/plots',
        'parameters': [
            ["xH1", 0, 1, r"$x_{H1}$"],
            ["xH2", 0, 1, r"$x_{H2}$"],
            ["xH3", 0, 1, r"$x_{H3}$"]
        ]
    }
}

# Load the pretrained flow model
model_params = {
    'flow': {
        'n_dim': 3,
        'n_blocks': 8, #6
        'n_nodes': 256,
        'cond_dims': 303,  # Adjust based on the flattened 2DPS input size
        'dropout': 0.0,
        'load': False,
        'model_location': 'trained_model.pth',
    }
}

flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.load_state_dict(torch.load(model_dir + '/trained_model.pth'))
flow_model.flow.eval()

# Create inference model instance
inference_model = InferenceModel(
    params=params,
    flow_model=flow_model,
    data_dir=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Run inference
inference_model.main()


