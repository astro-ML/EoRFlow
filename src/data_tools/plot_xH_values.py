import os
import numpy as np
import matplotlib.pyplot as plt

def load_labels_from_folders(folders):
    """
    Load the 'label' key from all .npz files in the given folders.
    
    Args:
        folders (list): List of directory paths containing .npz files.

    Returns:
        np.ndarray: Flattened array of all label values.
    """
    labels = []

    for folder in folders:
        npz_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        for file in npz_files:
            try:
                data = np.load(file)
                if 'label' in data:
                    labels.append(data['label'][:3].flatten())  # Flatten to ensure a 1D list
            except Exception as e:
                print(f"Error loading {file}: {e}")

    if labels:
        return np.concatenate(labels)  # Combine all labels into a single array
    else:
        return np.array([])

def plot_label_histogram(folders, bins=50):
    """
    Load labels from folders and plot a histogram.

    Args:
        folders (list): List of directories containing .npz files.
        bins (int): Number of bins in the histogram.
    """
    labels = load_labels_from_folders(folders)

    if labels.size == 0:
        print("No valid labels found to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(labels, bins=bins, edgecolor='black', alpha=0.75)
    plt.xlabel("Label Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Labels")
    plt.savefig('SKA_xH.pdf')
    plt.grid(True)
  

# Example usage:
folders = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2']
plot_label_histogram(folders)