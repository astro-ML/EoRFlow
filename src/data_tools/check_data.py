import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from data_loader import PowerSpectrumDataset, PowerSpectrumDataset_CNN, PowerSpectrumDataset_global
from sklearn.preprocessing import RobustScaler

def load_and_normalize_data(data_paths):
    dataset = PowerSpectrumDataset_global(data_paths)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Load data
    all_data = []
    for batch in data_loader:
        ps_batch, _ = batch
        all_data.append(ps_batch.numpy())
    
    all_data = np.concatenate(all_data, axis=0)

    # Reshape data for scaling
    all_data_flattened = all_data.reshape(-1, 1)

    # Apply robust scaling
    #scaler = RobustScaler()
    #all_data_scaled = scaler.fit_transform(all_data_flattened)

    # Reshape back to the original data shape
    #all_data = all_data_scaled.reshape(all_data.shape)
    

    # Normalize data

    # minmax
    #all_data = (all_data - np.min(all_data)) / (np.max(all_data) - np.min(all_data))

    

    # clipping
    #all_data = np.clip(all_data, 1e-6, None)

    # Now apply log10 safely
    #all_data = np.log10(all_data + 1e-6 )
    # zscore
    #all_data = (all_data - np.mean(all_data)) / np.std(all_data)
    
    #all_data = (all_data - np.min(all_data)) / (np.max(all_data) - np.min(all_data))
    #all_data = 1 / (1 + np.exp(-all_data))
    return all_data

def plot_distribution(data, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=100, density=False, alpha=0.75, color='blue')
    #plt.xlabel('Log10(Scaled Power Spectrum)')
    #plt.ylabel('Density')
    plt.title('Distribution Power Spectrum')
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()




# Paths to the data
train_data_path = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10'
#train_data_path_2 = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2'
val_data_path = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10'

# Load and normalize data
train_data = load_and_normalize_data([train_data_path])
val_data = load_and_normalize_data([val_data_path])

# Combine train and validation data
combined_data = np.concatenate((train_data, val_data), axis=0)
num_below_zero = (combined_data < 0).sum()

# Plot distribution
plot_distribution(combined_data, 'distribution_plot.pdf')




def check_negative_values_in_npz_files(folder_path):
    """
    This function checks if there are negative values in the 'image' key of all .npz files in a folder.

    Args:
        folder_path (str): Path to the folder containing .npz files.
    """
    # List all .npz files in the folder
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    # Loop through each file
    for file_name in npz_files:
        file_path = os.path.join(folder_path, file_name)

        # Load the .npz file
        try:
            data = np.load(file_path)

            # Check if 'image' key exists
            if 'image' in data:
                image = data['image']

                # Check for negative values in the 'image' key
                num_negative_values = (image < 0).sum()

                # Print the result for this file
                if num_negative_values > 0:
                    print(f"File {file_name} contains {num_negative_values} negative values in the 'image' key.")
                
            else:
                print(f"File {file_name} does not contain an 'image' key.")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Example usage
#folder_path = './1D_data/train'  # Replace this with the path to your folder containing .npz files
#check_negative_values_in_npz_files(folder_path)
