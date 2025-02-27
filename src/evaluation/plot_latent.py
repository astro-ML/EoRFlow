import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# ---------------------------
# Settings and paths
# ---------------------------
# Path to the trained model checkpoint and output directory for the plot.
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/SKA_flow/flow_8_256_weighted_pure'
trained_model_path = os.path.join(model_dir, 'trained_model.pth')
plot_output_path = os.path.join(model_dir, 'latent_forward_distribution.pdf')

# Test data directory (should contain test examples)
test_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10']

# ---------------------------
# Flow model parameters (should match the trained model)
# ---------------------------
model_params = {
    'flow': {
        'n_dim': 3,         # 3 xH parameters
        'n_blocks':8,
        'n_nodes': 256,
        'cond_dims': 303,   # Condition is the flattened 2DPS (3 channels * 10 * 10 = 300)
        'load': False,
        'model_location': 'trained_model.pth',
        'dropout': 0.3,
    }
}

# ---------------------------
# Logging setup (optional)
# ---------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Set device
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------
# Load the trained flow model
# ---------------------------
model = ConditionalInvertibleBlock(model_params)
model.flow.to(device)
model.flow.load_state_dict(torch.load(trained_model_path))
model.flow.eval()
logging.info("Trained flow model loaded.")

# ---------------------------
# Load test dataset and create DataLoader
# ---------------------------
test_dataset = PowerSpectrumDataset(test_data_dir)
# Here we assume that each sample from the dataset is a tuple: (ps_data, target)
# where ps_data is the condition (e.g., flattened 2DPS) and target is the true xH values.
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
logging.info(f"Test dataset size: {len(test_dataset)}")

# ---------------------------
# Pass test labels through the forward mapping to obtain latent variables
# ---------------------------
# We will loop over the test dataset, and for each sample we compute:
#    z, jac = model.flow(target, c=[ps_data], rev=False)
# If your data are not already in the expected shape (e.g., flattened), adjust as needed.
latent_list = []
for ps_data, target in test_loader:
    ps_data = ps_data.to(device).float()
    target = target.to(device).float()  # target: true xH values, shape: [1, 3]
    
    # (Optionally, if ps_data is not flattened, ensure it is in the correct shape)
    if len(ps_data.shape) > 2:
        ps_data = ps_data.view(ps_data.size(0), -1)  # flatten if needed

    with torch.no_grad():
        z, jac = model.flow(target, c=[ps_data], rev=False)
    latent_list.append(z.detach().cpu().numpy())

# Concatenate latent codes from all test samples.
latent_all = np.concatenate(latent_list, axis=0)  # shape: (num_samples, 3)
logging.info(f"Collected latent variables of shape: {latent_all.shape}")

# ---------------------------
# Plot histogram for each latent dimension
# ---------------------------
param_names = ['xH1_latent', 'xH2_latent', 'xH3_latent']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i in range(model_params['flow']['n_dim']):
    ax = axes[i]
    ax.hist(latent_all[:, i], bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f"Histogram of {param_names[i]}", fontsize=16)
    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    
    # Optionally, overlay a standard normal PDF for comparison.
    x_vals = np.linspace(-4, 4, 200)
    pdf_vals = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*x_vals**2)
    ax.plot(x_vals, pdf_vals, 'r--', label='Standard Normal')
    ax.legend(fontsize=12)
    ax.grid(True)

plt.tight_layout()
plt.savefig(plot_output_path)
plt.show()
logging.info(f"Latent space distribution histograms saved to {plot_output_path}")