import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
plt.style.use('seaborn-colorblind')

# Append paths to your modules if needed
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

# Import your models (adjust imports as necessary)
from cnn import CNN3D_film as CNN  # Replace with your actual CNN model import
from flow import ConditionalInvertibleBlock  # Replace with your actual flow model import

from torch.utils.data import Dataset, DataLoader

# Define the modified dataloader
class PowerSpectrumDatasetFromFiles(Dataset):
    def __init__(self, files, redshift_values=None):
        """
        Initialize the dataset with a list of file paths.

        Args:
            files (list): List of file paths to .npz files.
            redshift_values (array): Optional array of redshift values.
        """
        self.files = files

        # Load redshift values from the first file if not provided
        if redshift_values is None:
            first_file = self.files[0]
            data = np.load(first_file)
            self.redshift_values = data['redshifts']  # (30,)
            # Ensure redshift_values are in ascending order
            if self.redshift_values[0] > self.redshift_values[-1]:
                self.redshift_values = self.redshift_values[::-1]
        else:
            self.redshift_values = redshift_values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract the power spectrum and label
        ps = data['image']  # Shape (30, 10, 10)

        # Apply normalization
        ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)  # Z-score normalization
        ps = ps / (np.max(np.abs(ps)) + 1e-6)  # Scale to [-1, 1]

        label = data['label']  # xH values (30,)

        # Convert to PyTorch tensors
        ps_tensor = torch.tensor(ps, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return ps_tensor, label_tensor

# Define the InferenceModel class
class InferenceModel:
    """
    A class for inferring the xH values using a CNN + flow model, with 2DPS as input.
    """

    def __init__(self, params: dict, cnn_model: nn.Module, flow_model: nn.Module, dataset: Dataset, device: str = 'cpu') -> None:
        self.params = params['plot']
        self.cnn_model = cnn_model.to(device)
        self.flow_model = flow_model.flow.to(device)
        self.device = device

        # Use the provided dataset
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Set up output directory
        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        # Placeholder for storing CNN predictions and labels
        self.cnn_pred = None
        self.label = None
        self.n_labels = params['n_labels']
        self.redshifts = params['redshifts']
        self.cond_dims = params['cond_dims']

    def find_cnn_output(self, save_name: str = 'cnn_output_xH.npz'):
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

            # Redshift values
            redshifts = torch.tensor(self.redshifts / 10, dtype=torch.float32).to(self.device)

            for i, (ps_data, true_label) in enumerate(self.data_loader):
                ps_data, true_label = ps_data.to(self.device), true_label.to(self.device)

                ps_data = ps_data.unsqueeze(1)  # Add channel dimension

                redshift_batch = redshifts.repeat(ps_data.size(0), 1)  # shape: [batch_size, 30]

                # Forward through CNN
                cnn_output = self.cnn_model(ps_data, redshift_batch)

                # Concatenate CNN output and redshift information
                pred = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, cond_dims]

                self.cnn_pred[i] = pred.detach().cpu().numpy()
                self.label[i] = true_label.detach().cpu().numpy()

                if i % 100 == 0:
                    logging.info(f"Processed {i} samples.")

            np.savez(os.path.join(self.output_dir, save_name), cnn_pred=self.cnn_pred, label=self.label)

        return self.cnn_pred, self.label

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        """Calculate test statistics for inferred xH values, including coverage."""
        self.cnn_pred, self.label = self.find_cnn_output()
        logging.info('Calculating statistics for xH inference.')

        num_samples, num_params = self.label.shape
        mean = np.zeros((num_samples, num_params))
        lower = np.zeros((num_samples, num_params))
        upper = np.zeros((num_samples, num_params))
        rank = np.zeros((num_samples, num_params))

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
                logging.info(f"Processed {i} samples.")

        # Calculate empirical coverage percentages
        empirical_coverage = {level: (coverage_counts[level] / num_samples) * 100 for level in confidence_levels}

        # Save statistics including coverage
        np.savez(os.path.join(self.output_dir, output_name),
                 mean=mean, lower=lower, upper=upper, label=self.label, rank=rank, coverage=empirical_coverage)
        logging.info('Statistics saved with coverage.')

    def main(self) -> None:
        """Main function to run inference and calculate statistics."""
        self.calc_statistics(output_name='inference_statistics_xH.npz')
        logging.info("Inference and statistics calculation complete.")

# Define the function to check reionization completion
def is_reionization_finished(file_path, redshift_values, xH_threshold=0.01, z_target=5.0):
    """
    Check if reionization is finished by z_target in the given simulation.

    Args:
        file_path (str): Path to the .npz file containing the simulation data.
        redshift_values (array): Array of redshift values corresponding to xH values.
        xH_threshold (float): Threshold for xH to consider reionization finished.
        z_target (float): Redshift at which to check if reionization is finished.

    Returns:
        bool: True if reionization is finished by z_target, False otherwise.
    """
    data = np.load(file_path)
    xH = data['label']  # xH values (30,)

    # Ensure xH is ordered according to ascending redshift
    if redshift_values[0] > redshift_values[-1]:
        xH = xH[::-1]

    # Check that the lowest redshift is z=5
    if not np.isclose(redshift_values[0], 5.0):
        raise ValueError(f"The lowest redshift is not 5.0 in file {file_path}.")

    xH_at_z5 = xH[0]

    return xH_at_z5 <= xH_threshold

    
# Main script
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Paths to your data folders
    folder_no_noise = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10'
    folder_with_noise = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'

    # Paths to your models
    model_dir_no_noise = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pure_talk'  # Update with your path
    model_dir_with_noise = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_noise_talk'   # Update with your path

    # Load redshift values (adjust as necessary)
    redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
        7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
       10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
       12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
       15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
       17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])  # Replace with your actual redshift values

    # Step 1: List files in both folders
    files_no_noise = os.listdir(folder_no_noise)
    files_with_noise = os.listdir(folder_with_noise)

    print(f"Number of files in folder_no_noise: {len(files_no_noise)}")
    print(f"Number of files in folder_with_noise: {len(files_with_noise)}")

    # Find common files
    common_files = sorted(list(set(files_no_noise).intersection(set(files_with_noise))))
    logging.info(f'Number of common files: {len(common_files)}')

    if len(common_files) == 0:
        print("No common files found between the two datasets.")
        exit(1)

    # Generate the list of common files with full paths
    common_files_no_noise = [os.path.join(folder_no_noise, f) for f in common_files]
    common_files_with_noise = [os.path.join(folder_with_noise, f) for f in common_files]

    # Ensure redshift_values is loaded
    # Assuming redshift_values is already defined

    # Filter the common files to exclude simulations where reionization isn't finished by z=5
    filtered_common_files = []
    for file_no_noise, file_with_noise in zip(common_files_no_noise, common_files_with_noise):
        if is_reionization_finished(file_no_noise, redshift_values, xH_threshold=0.01, z_target=5.0):
            filtered_common_files.append((file_no_noise, file_with_noise))
        else:
            logging.info(f"Excluding simulation {os.path.basename(file_no_noise)} where reionization isn't finished by z=5.")

    # Unpack the filtered lists
    filtered_common_files_no_noise = [pair[0] for pair in filtered_common_files]
    filtered_common_files_with_noise = [pair[1] for pair in filtered_common_files]

    print(f"Number of simulations after filtering: {len(filtered_common_files_no_noise)}")

    # Proceed only if we have simulations after filtering
    if len(filtered_common_files_no_noise) == 0:
        print("No simulations left after filtering. Exiting.")
        exit(1)

    # Create datasets using the filtered lists
    dataset_no_noise = PowerSpectrumDatasetFromFiles(filtered_common_files_no_noise)
    dataset_with_noise = PowerSpectrumDatasetFromFiles(filtered_common_files_with_noise, redshift_values=dataset_no_noise.redshift_values)

    # Step 3: Load models
    # Load the pretrained CNN and flow models for no noise
    cnn_model_no_noise = CNN()
    cnn_model_no_noise.load_state_dict(torch.load(os.path.join(model_dir_no_noise, 'best_cnn_model.pth'), map_location='cpu'))
    cnn_model_no_noise.eval()

    model_params_no_noise = {
        'flow': {
            'n_dim': 30,  # Number of labels
            'n_blocks': 6,
            'n_nodes': 256,
            'cond_dims': 40,
            'dropout': 0.0,
            'load': False,
            'model_location': 'trained_model.pth',
        }
    }
    flow_model_no_noise = ConditionalInvertibleBlock(model_params_no_noise)
    flow_model_no_noise.flow.load_state_dict(torch.load(os.path.join(model_dir_no_noise, 'best_flow_model.pth'), map_location='cpu'))
    flow_model_no_noise.flow.eval()

    # Load the pretrained CNN and flow models for noise
    cnn_model_with_noise = CNN()
    cnn_model_with_noise.load_state_dict(torch.load(os.path.join(model_dir_with_noise, 'best_cnn_model.pth'), map_location='cpu'))
    cnn_model_with_noise.eval()

    model_params_with_noise = {
        'flow': {
            'n_dim': 30,  # Number of labels
            'n_blocks': 6,
            'n_nodes': 256,
            'cond_dims': 40,
            'dropout': 0.0,
            'load': False,
            'model_location': 'trained_model.pth',
        }
    }
    flow_model_with_noise = ConditionalInvertibleBlock(model_params_with_noise)
    flow_model_with_noise.flow.load_state_dict(torch.load(os.path.join(model_dir_with_noise, 'best_flow_model.pth'), map_location='cpu'))
    flow_model_with_noise.flow.eval()

    # Step 4: Define parameters
    params_no_noise = {
        'n_labels': 30,  # Number of xH values
        'redshifts': redshift_values,
        'cond_dims': 40,
        'plot': {
            'plot_dir': os.path.join(model_dir_no_noise, 'plots_no_noise'),
        }
    }

    params_with_noise = {
        'n_labels': 30,
        'redshifts': redshift_values,
        'cond_dims': 40,
        'plot': {
            'plot_dir': os.path.join(model_dir_with_noise, 'plots_with_noise'),
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 5: Run inference on both datasets
    inference_model_no_noise = InferenceModel(
        params=params_no_noise,
        cnn_model=cnn_model_no_noise,
        flow_model=flow_model_no_noise,
        dataset=dataset_no_noise,
        device=device
    )
    inference_model_no_noise.main()

    inference_model_with_noise = InferenceModel(
        params=params_with_noise,
        cnn_model=cnn_model_with_noise,
        flow_model=flow_model_with_noise,
        dataset=dataset_with_noise,
        device=device
    )
    inference_model_with_noise.main()

    # Step 6: Compare results
    data_no_noise = np.load(os.path.join(inference_model_no_noise.output_dir, 'inference_statistics_xH.npz'))
    mean_no_noise = data_no_noise['mean']
    lower_no_noise = data_no_noise['lower']
    upper_no_noise = data_no_noise['upper']
    label_no_noise = data_no_noise['label']

    data_with_noise = np.load(os.path.join(inference_model_with_noise.output_dir, 'inference_statistics_xH.npz'))
    mean_with_noise = data_with_noise['mean']
    lower_with_noise = data_with_noise['lower']
    upper_with_noise = data_with_noise['upper']
    label_with_noise = data_with_noise['label']

    # Verify that labels match
    for index in range(len(filtered_common_files_no_noise)):
        if not np.allclose(label_no_noise[index], label_with_noise[index]):
            logging.warning(f"Labels do not match for sample {index}")

    # Plot comparisons
    redshift_values = inference_model_no_noise.redshifts
    output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_mutual'  # Set your desired output directory
    os.makedirs(output_dir, exist_ok=True)

    for index in range(len(filtered_common_files_no_noise)):
        plt.figure(figsize=(10, 6))
        plt.fill_between(redshift_values, lower_with_noise[index], upper_with_noise[index], color='purple', alpha=0.3, label='68% CI With Noise')
        plt.plot(redshift_values, mean_with_noise[index], color='purple', label='Inferred Mean With Noise')
        plt.fill_between(redshift_values, lower_no_noise[index], upper_no_noise[index], color='orange', alpha=0.3, label='68% CI Without Noise')
        plt.plot(redshift_values, mean_no_noise[index], color='orange', label='Inferred Mean Without Noise')
        plt.plot(redshift_values, label_no_noise[index], color='black', linestyle='--', label='True xH')
        plt.xlabel('Redshift z', fontsize=16)
        plt.ylabel('Neutral Hydrogen Fraction xH', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.title(f'Reionization History Comparison for Sample {index + 1}', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        sample_filename = os.path.splitext(os.path.basename(filtered_common_files_no_noise[index]))[0]
        plt.savefig(os.path.join(output_dir, f'reionization_history_comparison_{sample_filename}.pdf'))
        plt.close()

    logging.info("Comparison plots saved.")

if __name__ == '__main__':
    main()
