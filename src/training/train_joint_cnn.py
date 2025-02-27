import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import logging

# Import your CNN and Flow model definitions and dataset.
from cnn import CNN3D_SKA as CNN   
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_SKA

# Hyperparameters
lr = 0.001
batch_size = 16
num_epochs = 1000

# Define output directory for models, logs, and plots.
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/SKA_CNN/CNN_6_256_weighed_BN_9'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging (logs will be saved in output_dir/train.log).
log_filename = os.path.join(output_dir, 'train.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # Overwrites the log file if it exists.
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Set device (GPU if available).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------
# Dataset & DataLoader Setup
# ---------------------------
data_train = [
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2'
]
# Optionally, if you want to compute importance weights, set a flag in your dataset.
# (Make sure your dataset class supports this by returning (ps_tensor, label_tensor, weight_tensor))
train_dataset = PowerSpectrumDataset_SKA(data_train, compute_weights=True)

# Define redshifts as condition (normalize using min-max for [0,1]).
z1, z2, z3 = 6.54, 7.19, 7.96

redshifts = torch.tensor([z1/10, z2/10, z3/10], dtype=torch.float32).to(device)  # shape: (3,)

# Split dataset into training and validation subsets.
train_ratio = 0.8
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

logging.info(f"Training set size: {len(train_subset)}")
logging.info(f"Validation set size: {len(val_subset)}")

# ---------------------------
# Model Initialization
# ---------------------------
# Initialize CNN model (which outputs 10 numbers) and move to device.
cnn_model = CNN().to(device)

# Flow model parameters: we infer 3 xH values and use a condition of dimension 13 (10 from CNN + 3 redshifts).
model_params = {
    'flow': {
        'n_dim': 3,        # Number of parameters to infer (e.g., 3 xH values)
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': 3+9,   # Condition is [CNN_output (10) concatenated with redshifts (3)]
        'load': False,
        'model_location': 'trained_model.pth',
        'dropout': 0.0,
    }
}
flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# ---------------------------
# Optimizers and Scheduler
# ---------------------------
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=lr, weight_decay=0)
flow_optimizer = optim.AdamW(flow_model.flow.parameters(), lr=lr, weight_decay=0)

# Scheduler for the flow optimizer (can extend to both if needed).
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    flow_optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# ---------------------------
# Loss Functions
# ---------------------------
def flow_loss(flow, y, cond, n_dim):
    """
    Standard flow loss: (0.5 * ||z||^2 - log_det) averaged over batch and normalized by n_dim.
    """
    z, jac = flow(y, c=[cond], rev=False)
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    return loss.mean() / n_dim

def weighted_flow_loss(flow, y, cond, n_dim, sample_weights):
    """
    Weighted flow loss, where each sample's loss is multiplied by its importance weight.
    sample_weights: tensor of shape [batch_size]
    """
    z, jac = flow(y, c=[cond], rev=False)
    losses = 0.5 * torch.sum(z ** 2, dim=1) - jac  # shape: [batch_size]
    weighted_losses = sample_weights * losses
    # Normalize by the sum of weights.
    loss = weighted_losses.sum() / sample_weights.sum() / n_dim
    return loss

# ---------------------------
# Training and Validation Loop (with reweighting)
# ---------------------------
def train_and_validate(cnn_model, flow_model, train_loader, val_loader, num_epochs):
    logging.info("Starting CNN+Flow training with importance reweighting...")
    best_val_loss = float('inf')
    patience = 20  # Early stopping patience.
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training Phase.
        cnn_model.train()
        flow_model.flow.train()
        for batch in train_loader:
            # Check if batch includes weights.
            if len(batch) == 3:
                ps_batch, target_batch, weight_batch = batch
                weight_batch = weight_batch.to(device)
            else:
                ps_batch, target_batch = batch
                weight_batch = torch.ones(ps_batch.size(0), device=device)
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)

            # For 3D CNN input, add an extra channel dimension.
            ps_batch = ps_batch.unsqueeze(1)

            # Create a batch of redshifts by repeating the normalized redshift vector.
            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)

            cnn_optimizer.zero_grad()
            flow_optimizer.zero_grad()

            # Forward pass through CNN: output shape should be (batch_size, 10)
            cnn_output = cnn_model(ps_batch, redshift_batch)
            # Concatenate CNN output (10) with redshift (3) -> condition has shape (batch_size, 13)
            condition = torch.cat([cnn_output, redshift_batch], dim=1)

            # Compute weighted loss through the flow model using the condition.
            loss = weighted_flow_loss(flow=flow_model.flow, y=target_batch, cond=condition,
                                      n_dim=model_params['flow']['n_dim'], sample_weights=weight_batch)
            loss.backward()

            # Gradient clipping for stability.
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)

            cnn_optimizer.step()
            flow_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase.
        cnn_model.eval()
        flow_model.flow.eval()
        with torch.no_grad():
            for batch in val_loader:
                # In validation, we use uniform weights.
                if len(batch) == 3:
                    ps_batch, target_batch, _ = batch
                else:
                    ps_batch, target_batch = batch
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1)
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
                cnn_output = cnn_model(ps_batch, redshift_batch)
                condition = torch.cat([cnn_output, redshift_batch], dim=1)
                # Use a tensor of ones as weights.
                uniform_weights = torch.ones(ps_batch.size(0), device=device)
                val_loss = weighted_flow_loss(flow=flow_model.flow, y=target_batch, cond=condition,
                                              n_dim=model_params['flow']['n_dim'], sample_weights=uniform_weights)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, "
                     f"Validation Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)
        current_lr = flow_optimizer.param_groups[0]['lr']
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'best_cnn_model.pth'))
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'best_flow_model.pth'))
            logging.info(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered.")
                break

    # Reload best models.
    cnn_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_cnn_model.pth')))
    flow_model.flow.load_state_dict(torch.load(os.path.join(output_dir, 'best_flow_model.pth')))
    return train_losses, val_losses

# ---------------------------
# Run Training
# ---------------------------
train_losses, val_losses = train_and_validate(cnn_model, flow_model, train_loader, val_loader, num_epochs)

# Save final models.
torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'flow_model.pth'))
logging.info(f"Final models saved to {output_dir}")

# Save loss histories.
np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))
logging.info("Training and validation losses saved as .npy files")

# Plot and save training curves.
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN+Flow Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'cnn_flow_training_validation_loss.pdf'))
logging.info("Training curves saved.")