import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

# Make sure to import your dataset class. Adjust paths as needed.
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
from data_loader import PowerSpectrumDataset

# Set up logging (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories containing your .npz files (adjust as needed)
data_dirs = [
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10',
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2'
]

# Create dataset with importance sampling enabled
# skip_boundary_labels: whether to drop boundary samples (True/False)
# skip_fraction: fraction of boundary samples to drop (0 to 1)
# compute_weights: True -> compute importance weights based on target distribution
dataset = PowerSpectrumDataset(data_dirs, compute_weights=True)

# Iterate through the dataset to collect labels and weights.
labels = []
weights = []
num_samples = len(dataset)
logging.info(f"Processing {num_samples} samples for label/weight plots.")

for idx in range(num_samples):
    # Depending on compute_weights flag, __getitem__ returns (ps_tensor, label_tensor, weight)
    sample = dataset[idx]
    if len(sample) == 3:
        _, label_tensor, weight_tensor = sample
        weight = weight_tensor.item()
    else:
        _, label_tensor = sample
        weight = 1.0
    # Assuming label_tensor is 1D (for 1 xH value) or a small vector.
    # Here, we'll compute a scalar summary (e.g. mean) if necessary.
    label_np = label_tensor.detach().cpu().numpy().flatten()
    # For a single xH value, this is just the value.
    # For multiple dimensions, you might want to plot each dimension separately.
    label_scalar = label_np.mean()
    labels.append(label_scalar)
    weights.append(weight)

labels = np.array(labels)
weights = np.array(weights)

# Plot histogram of label distribution.
plt.figure(figsize=(8, 6))
plt.hist(labels, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
plt.xlabel("Label value")
plt.ylabel("Count")
plt.title("Histogram of Labels")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'label_distribution.pdf'))
plt.show()

# Plot histogram of importance weights.
plt.figure(figsize=(8, 6))
plt.hist(weights, bins=50, alpha=0.7, color='seagreen', edgecolor='black')
plt.xlabel("Importance Weight")
plt.ylabel("Count")
plt.title("Histogram of Importance Weights")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'weight_distribution.pdf'))
plt.show()

# Plot a scatter plot of label vs. weight.
plt.figure(figsize=(8, 6))
plt.scatter(labels, weights, alpha=0.5, color='darkorange')
plt.xlabel("Label Value")
plt.ylabel("Importance Weight")
plt.title("Scatter Plot: Label vs. Importance Weight")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'label_vs_weight.pdf'))
plt.show()