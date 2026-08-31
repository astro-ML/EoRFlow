import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

# Import the data loader class
from torch.utils.data import Dataset
from data_loader import PowerSpectrumDataset_global_sample as PowerSpectrumDataset


# Functions to compute reionization metrics
def compute_reionization_end_redshift(redshift_values, xH_values, threshold=0.01):
    # Ensure redshift_values are in ascending order (from low z to high z)
    if redshift_values[0] > redshift_values[-1]:
        redshift_values = redshift_values[::-1]
        xH_values = xH_values[:, ::-1]

    reionization_end_redshifts = []
    for xH in xH_values:
        idx = np.where(xH <= threshold)[0]
        if idx.size > 0:
            z_end = redshift_values[idx[-1]]  # Use the highest redshift where xH <= threshold
        else:
            z_end = redshift_values[0]  # Reionization not ended within the redshift range
        reionization_end_redshifts.append(z_end)
    return np.array(reionization_end_redshifts)

def compute_reionization_midpoint_redshift(redshift_values, xH_values):
    reionization_mid_redshifts = []
    for xH in xH_values:
        idx = np.where(xH <= 0.5)[0]
        if idx.size > 0:
            z_mid = redshift_values[idx[-1]]  # Use the highest redshift where xH <= 0.5
        else:
            z_mid = redshift_values[0]  # Midpoint not reached within the redshift range
        reionization_mid_redshifts.append(z_mid)
    return np.array(reionization_mid_redshifts)

def compute_reionization_duration(redshift_values, xH_values, start_threshold=0.99, end_threshold=0.01):
    durations = []
    for xH in xH_values:
        idx_start = np.where(xH <= start_threshold)[0]
        idx_end = np.where(xH <= end_threshold)[0]
        if idx_start.size > 0 and idx_end.size > 0:
            z_start = redshift_values[idx_start[-1]]  # Highest redshift where xH <= start_threshold
            z_end = redshift_values[idx_end[-1]]      # Highest redshift where xH <= end_threshold
            duration = z_end - z_start
        else:
            duration = np.nan  # Unable to compute duration
        durations.append(duration)
    return np.array(durations)

def count_unfinished_reionization_at_z5(redshift_values, xH_values, xH_threshold=0.1, z_target=5.0):
    """
    Counts and prints the number of simulations where reionization has not finished
    by redshift z = 5, meaning xH > xH_threshold at z = z_target.

    Args:
        redshift_values (np.ndarray): Array of redshift values.
        xH_values (np.ndarray): Array of xH values for all samples, shape (num_samples, num_redshifts).
        xH_threshold (float): Threshold for xH to consider reionization unfinished.
        z_target (float): The redshift at which to check xH values.
    """
    # Find the index in redshift_values closest to z_target
    idx_z = np.argmin(np.abs(redshift_values - z_target))
    z_actual = redshift_values[idx_z]

    # Extract xH values at z_target
    xH_at_z = xH_values[:, idx_z]

    # Count the number of simulations where xH > xH_threshold at z_target
    num_unfinished = np.sum(xH_at_z > xH_threshold)
    total_samples = xH_values.shape[0]
    percentage_unfinished = 100 * num_unfinished / total_samples

    print(f"At redshift z = {z_actual:.2f}:")
    print(f"Number of simulations with xH > {xH_threshold}: {num_unfinished}/{total_samples} ({percentage_unfinished:.2f}%)")
    print(f"Number of simulations with xH ≤ {xH_threshold}: {total_samples - num_unfinished}/{total_samples} ({100 - percentage_unfinished:.2f}%)")



"""
def main():
    # Define data directories
    #data_dirs=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10']
    data_dirs = [
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/toms_data_pure']
    #data_dirs = [
    #'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise',
    #'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise_astro']

    # Initialize the dataset
    dataset = PowerSpectrumDataset(data_dirs, exclude_unfinished_reionization=False, exclude_early_reionization=False, undersample_xH=True)

    # Since redshift values are assumed to be the same for all samples, retrieve them from the first file
    first_file = dataset.files[0]
    data = np.load(first_file)
    redshift_values = data['redshifts']  # (30,)
    print(redshift_values.shape)
    # Ensure redshift_values are in ascending order
    if redshift_values[0] > redshift_values[-1]:
        redshift_values = redshift_values[::-1]

    # Initialize list to store xH values
    xH_values_list = []

    # Iterate over the dataset to collect xH values
    for i, file_path in enumerate(dataset.files):
        data = np.load(file_path)
        label = data['label']  # xH values (30,)
        xH = label.copy()
        # Ensure xH is ordered according to ascending redshift
        if redshift_values[0] > redshift_values[-1]:
            xH = xH[::-1]
        xH_values_list.append(xH)
        if i % 1000 == 0:
            print(f"Processed {i} samples")

    # Convert list to NumPy array
    xH_values = np.array(xH_values_list)  # Shape: (num_samples, num_redshifts)

    # Now, flatten all xH values into a 1D array
    all_xH_values = xH_values.flatten()

    # Plot histogram of all true xH values
    plt.figure(figsize=(10, 6))
    plt.hist(all_xH_values, bins=100, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('True xH Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of All True xH Values in Test Dataset', fontsize=14)
    plt.grid(True)
    plt.savefig('true_xH_values_histogram.png')
    plt.show()

    
    # Now, compute the reionization metrics
    #reionization_end_redshifts = compute_reionization_end_redshift(redshift_values, xH_values)
    #reionization_mid_redshifts = compute_reionization_midpoint_redshift(redshift_values, xH_values)
    #reionization_durations = compute_reionization_duration(redshift_values, xH_values)

    # Plot histograms

    

    # ----------------------------------------------------
    # Plot of All Reionization Histories (xH as a function of z)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))

    num_samples = xH_values.shape[0]

    # Option 1: Plot all reionization histories with low opacity
    for xH in xH_values:
        plt.plot(redshift_values, xH, color='blue', alpha=0.01)

    # Option 2: Plot a random subset if the dataset is too large
    # num_to_plot = 1000
    # indices = np.random.choice(num_samples, size=num_to_plot, replace=False)
    # for idx in indices:
    #     plt.plot(redshift_values, xH_values[idx], color='blue', alpha=0.1)

    plt.xlabel('Redshift z', fontsize=15)
    plt.ylabel('Neutral Hydrogen Fraction xH', fontsize=15)
    plt.title('Reionization Histories of Training data', fontsize=16)
    # Do not invert x-axis; redshift increases from left to right
    plt.grid(True)
    plt.savefig('reionization_histories.pdf')
    plt.show()
    
if __name__ == '__main__':
    main()    
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_xH_labels(folders):
    """
    Compute the mean xH label for each sample in the given folders.

    Args:
        folders (list): List of directory paths containing .npz files.

    Returns:
        list: Mean xH values for all samples.
    """
    mean_xH_values = []

    for folder in folders:
        npz_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        for file in npz_files:
            try:
                data = np.load(file)
                if 'label' in data:
                    mean_xH = data['label']  # Compute mean of xH values
                    mean_xH_values.append(mean_xH.flatten())
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return mean_xH_values

def plot_xH_histogram(folders, bins=50):
    """
    Load mean xH values from folders and plot a histogram.

    Args:
        folders (list): List of directories containing .npz files.
        bins (int): Number of bins in the histogram.
    """
    mean_xH_values = compute_mean_xH_labels(folders)

    if not mean_xH_values:
        print("No valid xH values found to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(mean_xH_values, bins=bins, edgecolor='black', alpha=0.75)
    plt.xlabel("Mean xH Label")
    plt.ylabel("Frequency")
    plt.title("Histogram of Mean xH Labels")
    plt.grid(True)
    plt.savefig('mean_xH_sample.png')

# Example usage:
folders = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/toms_data_pure']
plot_xH_histogram(folders)


























