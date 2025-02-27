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
model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/SKA_flow/flow_8_256_weighted_pure'
trained_model_path = os.path.join(model_dir, 'trained_model.pth')
plot_output_path = os.path.join(model_dir, 'jacobian_logdet_histogram.pdf')

# Test data directory (ensure it contains your test examples)
test_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10']

# ---------------------------
# Flow model parameters (should match the trained model)
# ---------------------------
model_params = {
    'flow': {
        'n_dim': 3,         # 3 xH parameters
        'n_blocks': 8,
        'n_nodes': 256,
        'cond_dims': 303,   # e.g., 3*10*10 = 300 for the flattened 2D power spectrum.
        'load': False,
        'model_location': 'trained_model.pth',
        'dropout': 0.0,
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
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
logging.info(f"Test dataset size: {len(test_dataset)}")

# ---------------------------
# Collect log-determinant (Jacobian) values
# ---------------------------
log_det_list = []

# Loop over the test dataset. For each sample, pass the true label through the flow.
# The flow should return both the latent variable 'z' and the log-det of the Jacobian.
with torch.no_grad():
    for ps_data, target in test_loader:
        ps_data = ps_data.to(device).float()
        target = target.to(device).float()  # target should be the true xH values (shape: [1, 3])
        
        # If ps_data is not already flattened, flatten it as expected by your model:
        if len(ps_data.shape) > 2:
            ps_data = ps_data.view(ps_data.size(0), -1)
        
        z, jac = model.flow(target, c=[ps_data], rev=False)
        # Append the log-det values (convert to numpy)
        log_det_list.append(jac.detach().cpu().numpy())

# Concatenate all log-det values into a single array.
log_det_all = np.concatenate(log_det_list, axis=0)  # shape: (num_samples,)
logging.info(f"Collected log-det values of shape: {log_det_all.shape}")

# ---------------------------
# Plot histogram of log-determinant values
# ---------------------------
plt.figure(figsize=(8, 6))
plt.hist(log_det_all, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel("Log-Determinant", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Histogram of Log-Determinant of the Jacobian", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_output_path)
plt.show()
logging.info(f"Jacobian log-determinant histogram saved to {plot_output_path}")