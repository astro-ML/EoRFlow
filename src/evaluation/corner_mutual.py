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
from getdist import plots, MCSamples

class PowerSpectrumDatasetFromFiles(Dataset):
    def __init__(self, files, redshift_values=None):
        self.files = files
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
        ps = data['image']  # Shape (30, 10, 10)

        # Normalize
        ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)
        ps = ps / (np.max(np.abs(ps)) + 1e-6)

        label = data['label']  # (30,)

        ps_tensor = torch.tensor(ps, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return ps_tensor, label_tensor

class InferenceModel:
    def __init__(self, params: dict, cnn_model: nn.Module, flow_model: nn.Module, dataset: Dataset, device: str = 'cpu') -> None:
        self.params = params['plot']
        self.cnn_model = cnn_model.to(device)
        self.flow_model = flow_model.flow.to(device)
        self.device = device

        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.output_dir = self.params['plot_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.cnn_pred = None
        self.label = None
        self.n_labels = params['n_labels']
        self.redshifts = params['redshifts']
        self.cond_dims = params['cond_dims']

    def find_cnn_output(self, save_name: str = 'cnn_output_xH.npz'):
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

            redshifts = torch.tensor(self.redshifts / 10, dtype=torch.float32).to(self.device)

            for i, (ps_data, true_label) in enumerate(self.data_loader):
                ps_data, true_label = ps_data.to(self.device), true_label.to(self.device)
                ps_data = ps_data.unsqueeze(1)

                redshift_batch = redshifts.repeat(ps_data.size(0), 1)
                cnn_output = self.cnn_model(ps_data, redshift_batch)

                pred = torch.cat([cnn_output, redshift_batch], dim=1)
                self.cnn_pred[i] = pred.detach().cpu().numpy()
                self.label[i] = true_label.detach().cpu().numpy()

                if i % 100 == 0:
                    logging.info(f"Processed {i} samples.")

            np.savez(os.path.join(self.output_dir, save_name), cnn_pred=self.cnn_pred, label=self.label)

        return self.cnn_pred, self.label

    def calc_statistics(self, output_name: str = 'inference_statistics_xH.npz', sample_size: int = 1000) -> None:
        self.cnn_pred, self.label = self.find_cnn_output()
        logging.info('Calculating statistics for xH inference.')

        num_samples, num_params = self.label.shape
        mean = np.zeros((num_samples, num_params))
        lower = np.zeros((num_samples, num_params))
        upper = np.zeros((num_samples, num_params))
        rank = np.zeros((num_samples, num_params))

        confidence_levels = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        coverage_counts = {level: np.zeros(num_params) for level in confidence_levels}

        for i in range(num_samples):
            z = torch.randn((sample_size, self.n_labels)).to(self.device)
            condition = torch.Tensor(self.cnn_pred[i]).unsqueeze(0).repeat(sample_size, 1).to(self.device)
            samples, _ = self.flow_model(z, c=[condition], rev=True)
            samples = samples.detach().cpu().numpy()

            mean[i] = np.mean(samples, axis=0)
            lower[i] = np.percentile(samples, 16, axis=0)
            upper[i] = np.percentile(samples, 84, axis=0)

            for j in range(num_params):
                rank[i, j] = np.sum(samples[:, j] < self.label[i, j])

            true_values = self.label[i]
            for level in confidence_levels:
                lower_bound = np.percentile(samples, (100 - level) / 2, axis=0)
                upper_bound = np.percentile(samples, 100 - (100 - level) / 2, axis=0)
                coverage = (true_values >= lower_bound) & (true_values <= upper_bound)
                coverage_counts[level] += coverage.astype(int)

            if i % 100 == 0:
                logging.info(f"Processed {i} samples.")

        empirical_coverage = {level: (coverage_counts[level] / num_samples) * 100 for level in confidence_levels}

        np.savez(os.path.join(self.output_dir, output_name),
                 mean=mean, lower=lower, upper=upper, label=self.label, rank=rank, coverage=empirical_coverage)
        logging.info('Statistics saved with coverage.')

    def plot_corner_for_redshifts(self, index: int, redshifts_of_interest: list, sample_size: int = 1000):
        # Find parameter indices for chosen redshifts
        param_indices = []
        for z in redshifts_of_interest:
            z_idx = np.argmin(np.abs(self.redshifts - z))
            param_indices.append(z_idx)

        self.cnn_pred, self.label = self.find_cnn_output()

        condition_vec = torch.tensor(self.cnn_pred[index], dtype=torch.float32).unsqueeze(0).to(self.device)
        z = torch.randn((sample_size, self.n_labels)).to(self.device)
        samples, _ = self.flow_model(z, c=[condition_vec.repeat(sample_size, 1)], rev=True)
        samples = samples.detach().cpu().numpy()

        subset_samples = samples[:, param_indices]
        param_names = [f"xH(z={int(r)})" for r in redshifts_of_interest]
        param_labels = [f"xH(z={r})" for r in redshifts_of_interest]

        mc_samples = MCSamples(samples=subset_samples, names=param_names, labels=param_labels)

        g = plots.get_subplot_plotter()
        g.triangle_plot([mc_samples], filled=True)

        true_values = self.label[index]
        for i, z_idx in enumerate(param_indices):
            g.subplots[i, i].axvline(true_values[z_idx], color='red', linestyle='--')

        for i in range(len(param_indices)-1):
            for j in range(i+1, len(param_indices)):
                ax = g.subplots[j, i]
                ax.plot(true_values[param_indices[i]], true_values[param_indices[j]], 'ro', markersize=6)

        plot_path = os.path.join(self.output_dir, f'corner_plot_sample_{index+1}_z{redshifts_of_interest}.pdf')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Corner plot saved for sample {index + 1} and redshifts {redshifts_of_interest}.")


    def main(self) -> None:
        self.calc_statistics(output_name='inference_statistics_xH.npz')
        logging.info("Inference and statistics calculation complete.")


def is_reionization_finished(file_path, redshift_values, xH_threshold=0.01, z_target=5.0):
    data = np.load(file_path)
    xH = data['label']  # xH values (30,)
    if redshift_values[0] > redshift_values[-1]:
        xH = xH[::-1]
    if not np.isclose(redshift_values[0], 5.0):
        raise ValueError(f"The lowest redshift is not 5.0 in file {file_path}.")
    xH_at_z5 = xH[0]
    return xH_at_z5 <= xH_threshold

def plot_joint_corner(inference_model_no_noise, inference_model_with_noise, index: int, redshifts_of_interest: list, sample_size: int = 1000):
    """
    Plot a single corner plot comparing posterior samples from the no-noise and with-noise models
    for the same sample (index) and the same set of chosen redshifts.
    """

    # Helper function to get samples from a given inference model
    def get_samples_for_model(inference_model, index, sample_size):
        # Ensure CNN predictions and labels are loaded
        inference_model.cnn_pred, inference_model.label = inference_model.find_cnn_output()

        # Condition vector for chosen sample
        condition_vec = torch.tensor(inference_model.cnn_pred[index], dtype=torch.float32).unsqueeze(0).to(inference_model.device)
        z = torch.randn((sample_size, inference_model.n_labels)).to(inference_model.device)
        samples, _ = inference_model.flow_model(z, c=[condition_vec.repeat(sample_size, 1)], rev=True)
        return samples.detach().cpu().numpy()

    # Get parameter indices for chosen redshifts
    param_indices = [np.argmin(np.abs(inference_model_no_noise.redshifts - z)) for z in redshifts_of_interest]

    # Get samples from both models
    samples_no_noise = get_samples_for_model(inference_model_no_noise, index, sample_size)
    samples_with_noise = get_samples_for_model(inference_model_with_noise, index, sample_size)

    # Subset the samples to the chosen parameters
    subset_no_noise = samples_no_noise[:, param_indices]
    subset_with_noise = samples_with_noise[:, param_indices]

    # Parameter names and labels
    param_names = [f"xH(z={int(r)})" for r in redshifts_of_interest]
    param_labels = [f"xH(z={r})" for r in redshifts_of_interest]

    # Create MCSamples objects
    mc_samples_no_noise = MCSamples(samples=subset_no_noise, names=param_names, labels=param_labels)
    mc_samples_with_noise = MCSamples(samples=subset_with_noise, names=param_names, labels=param_labels)

    # Create corner plot
    g = plots.get_subplot_plotter()
    g.settings.solid_colors = ["darkorange", "purple"]
    g.triangle_plot([mc_samples_with_noise, mc_samples_no_noise],
                    filled=True,
                    legend_labels=["With Noise", "No Noise"],
                    colors=["purple", "darkorange"])

    # Add true values as vertical/horizontal lines if desired
    true_values = inference_model_no_noise.label[index]
    for i, p_idx in enumerate(param_indices):
        g.subplots[i, i].axvline(true_values[p_idx], color='black', linestyle='--')

    for i in range(len(param_indices)-1):
        for j in range(i+1, len(param_indices)):
            ax = g.subplots[j, i]
            # Mark true values with a black dot
            ax.plot(true_values[param_indices[i]], true_values[param_indices[j]], 'ko', markersize=6)

    # Save the plot
    output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_mutual_paper'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'corner_plot_joint_sample_{index+1}_z{redshifts_of_interest}.pdf')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Joint corner plot saved for sample {index + 1} and redshifts {redshifts_of_interest}.")


def main():
    logging.basicConfig(level=logging.INFO)

    folder_no_noise = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10'
    folder_with_noise = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'

    model_dir_no_noise = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pure_talk'
    model_dir_with_noise = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_noise_talk'

    redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
        7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
       10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
       12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
       15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
       17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])

    files_no_noise = os.listdir(folder_no_noise)
    files_with_noise = os.listdir(folder_with_noise)

    common_files = sorted(list(set(files_no_noise).intersection(set(files_with_noise))))
    logging.info(f'Number of common files: {len(common_files)}')

    common_files_no_noise = [os.path.join(folder_no_noise, f) for f in common_files]
    common_files_with_noise = [os.path.join(folder_with_noise, f) for f in common_files]

    filtered_common_files = []
    for file_no_noise, file_with_noise in zip(common_files_no_noise, common_files_with_noise):
        if is_reionization_finished(file_no_noise, redshift_values, xH_threshold=0.01, z_target=5.0):
            filtered_common_files.append((file_no_noise, file_with_noise))
        else:
            logging.info(f"Excluding simulation {os.path.basename(file_no_noise)} where reionization isn't finished by z=5.")

    filtered_common_files_no_noise = [pair[0] for pair in filtered_common_files]
    filtered_common_files_with_noise = [pair[1] for pair in filtered_common_files]

    if len(filtered_common_files_no_noise) == 0:
        print("No simulations left after filtering. Exiting.")
        exit(1)

    dataset_no_noise = PowerSpectrumDatasetFromFiles(filtered_common_files_no_noise)
    dataset_with_noise = PowerSpectrumDatasetFromFiles(filtered_common_files_with_noise, redshift_values=dataset_no_noise.redshift_values)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params_no_noise = {
        'n_labels': 30,
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

    cnn_model_no_noise = CNN()
    cnn_model_no_noise.load_state_dict(torch.load(os.path.join(model_dir_no_noise, 'best_cnn_model.pth'), map_location='cpu'))
    cnn_model_no_noise.eval()

    model_params_no_noise = {
        'flow': {
            'n_dim': 30,
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

    cnn_model_with_noise = CNN()
    cnn_model_with_noise.load_state_dict(torch.load(os.path.join(model_dir_with_noise, 'best_cnn_model.pth'), map_location='cpu'))
    cnn_model_with_noise.eval()

    model_params_with_noise = {
        'flow': {
            'n_dim': 30,
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

    inference_model_no_noise = InferenceModel(params_no_noise, cnn_model_no_noise, flow_model_no_noise, dataset_no_noise, device)
    inference_model_no_noise.main()

    inference_model_with_noise = InferenceModel(params_with_noise, cnn_model_with_noise, flow_model_with_noise, dataset_with_noise, device)
    inference_model_with_noise.main()

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

    for index in range(len(filtered_common_files_no_noise)):
        if not np.allclose(label_no_noise[index], label_with_noise[index]):
            logging.warning(f"Labels do not match for sample {index}")

    output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_mutual_paper'
    os.makedirs(output_dir, exist_ok=True)

    """
    # Plot reionization history comparisons
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
    """

    # Plot corner plots for a few samples and redshifts of interest
    redshifts_of_interest = [6,8,10,12]  # example choice
    # Plot corner for first 5 samples for both no noise and with noise, for example
    for index in range(min(5, len(filtered_common_files_no_noise))):
        # Corner plot without noise
        inference_model_no_noise.plot_corner_for_redshifts(index=index, redshifts_of_interest=redshifts_of_interest, sample_size=1000)
        # Corner plot with noise
        inference_model_with_noise.plot_corner_for_redshifts(index=index, redshifts_of_interest=redshifts_of_interest, sample_size=1000)

    logging.info("Corner plots saved.")



    # Example usage after inference is done
    redshifts_of_interest = [6,8,10,12,14,16]
    for i in range(min(10, len(filtered_common_files_no_noise))):
        plot_joint_corner(inference_model_no_noise, inference_model_with_noise, index=i, redshifts_of_interest=redshifts_of_interest, sample_size=1000)


if __name__ == '__main__':
    main()
