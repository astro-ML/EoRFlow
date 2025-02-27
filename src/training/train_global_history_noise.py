import os
import sys
# Update your paths if necessary
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import logging

# Import the modified CNN model
from cnn import CNN3D_15 as CNN  
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global as PowerSpectrumDataset

lr = 0.001
batch_size = 16 # 16
num_epochs = 1000

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR/full_EoR_pure_z12_weights_dim'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
log_filename = os.path.join(output_dir, 'train.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize dataset
# pure
data_train=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10', 
            '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/toms_data_pure'] 
# noise
#data_train = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise', 
#'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise_astro']
# both
#data_train = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10',
#'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise', 
#'/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise_astro']
train_dataset = PowerSpectrumDataset(data_train, exclude_unfinished_reionization=False, exclude_early_reionization=False, compute_weights=True)

# Adjust redshift values as needed
# For example, if you have 10 redshift slices, adjust accordingly
#redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
#        7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
#       10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
#       12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
#       15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
#       17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])
#redshifts = torch.tensor(redshift_values / 10, dtype=torch.float32).to(device)  # Normalize if needed

# Split dataset into train and validation
train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Initialize CNN model and Flow model
input_size = (15, 10, 10)  # Updated input size
cnn_model = CNN().to(device)

model_params = {
    'flow': {
        'n_dim': 15,  # Inferring 30 xH values (adjust if needed)
        'n_blocks': 6, # 6
        'n_nodes': 256, # 256
        'cond_dims': 15+10,  # CNN output size (10) 
        'load': False,
        'model_location': 'trained_model.pth',
        'dropout': 0.0,
        'sigmoid': False
    }
}
flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# Set up optimizer
optimizer = optim.AdamW(
    list(cnn_model.parameters()) + list(flow_model.flow.parameters()),
    lr=lr, weight_decay=1e-5
)


def flow_loss(flow, y, cond, n_dim, sample_weights, num_slices=15):
    """
    Compute the weighted negative log-likelihood loss for a flow model,
    applying per-slice weights to the entire lightcone.

    Args:
        flow: The flow model.
        y (torch.Tensor): Target tensor of shape [batch_size, n_dim].
        cond (torch.Tensor): Conditioning variable.
        n_dim (int): Total dimensionality of the lightcone.
        sample_weights (torch.Tensor): Tensor of shape [batch_size, num_slices]
                                       containing per-slice weights.
        num_slices (int): Number of redshift slices in the lightcone.

    Returns:
        torch.Tensor: The weighted loss.
    """
    # Forward pass through the flow.
    z, jac = flow(y, c=[cond], rev=False)  # z: [batch_size, n_dim], jac: [batch_size]
    batch_size = z.shape[0]

    slice_dim = n_dim // num_slices  # e.g. 100 if n_dim = 1500 and num_slices = 15
    
    # Reshape the latent variable into its redshift slices.
    # Now z_slices has shape [batch_size, num_slices, slice_dim].
    z_slices = z.view(batch_size, num_slices, slice_dim)

    # Compute per-slice losses:
    # For each slice, sum over its latent dimensions.
    # Distribute the Jacobian term equally among the slices.
    losses_per_slice = 0.5 * torch.sum(z_slices ** 2, dim=2) - (jac.unsqueeze(1) / num_slices)

    # losses_per_slice now has shape [batch_size, num_slices].

    # Multiply each slice's loss by its corresponding weight.
    weighted_losses = sample_weights * losses_per_slice

    # Normalize: sum the weighted losses, divide by the sum of weights, and optionally scale by slice_dim.
    loss = weighted_losses.sum() / sample_weights.sum() / slice_dim
    return loss


# Define flow loss function
def unweighted_flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss


def train_and_validate(cnn_model, flow_model, train_loader, val_loader, num_epochs):
    logging.info("Starting CNN+Flow training...")
    cnn_model.train()
    flow_model.flow.train()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 20
    epochs_without_improvement = 0

    # Initialize optimizers and schedulers
    cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=lr, weight_decay=1e-5)
    flow_optimizer = optim.AdamW(flow_model.flow.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=flow_optimizer, mode='min', factor=0.5, patience=10
    )

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        cnn_model.train()
        flow_model.flow.train()
        for ps_batch, target_batch, redshift_batch, weight_batch in train_loader:
            ps_batch, target_batch, redshift_batch, weight_batch = ps_batch.to(device), target_batch.to(device), redshift_batch.to(device), weight_batch.to(device)
            
            ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
    
            cnn_optimizer.zero_grad()
            flow_optimizer.zero_grad()

            #redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # shape: [batch_size, redshift_dim]
            
            # Forward through CNN
            cnn_output = cnn_model(ps_batch, redshift_batch)
            
            # Concatenate redshift values with CNN output
            condition = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, cond_dims]

            # Forward through Flow using CNN output as condition
            loss = flow_loss(
                flow=flow_model.flow,
                y=target_batch,
                cond=condition,
                n_dim=model_params['flow']['n_dim'],
                sample_weights=weight_batch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            cnn_optimizer.step()
            flow_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        cnn_model.eval()
        flow_model.flow.eval()
        with torch.no_grad():
            for ps_batch, target_batch, redshift_batch, weight_batch in val_loader:
                ps_batch, target_batch, redshift_batch, weight_batch = ps_batch.to(device), target_batch.to(device), redshift_batch.to(device), weight_batch.to(device)

                ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN

                #redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # shape: [batch_size, redshift_dim]
            
                # Forward through CNN
                cnn_output = cnn_model(ps_batch, redshift_batch)
                
                # Concatenate redshift values with CNN output
                condition = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, cond_dims]

                # Forward through Flow
                val_loss = flow_loss(
                    flow=flow_model.flow,
                    y=target_batch,
                    cond=condition,
                    n_dim=model_params['flow']['n_dim'],
                    sample_weights=weight_batch
                )
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
        )

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'best_cnn_model.pth'))
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'best_flow_model.pth'))
            logging.info(f"Saved new best model at epoch {epoch+1} with validation loss {avg_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered.")
                break

    return train_losses, val_losses


# Train and save both models
train_losses, val_losses = train_and_validate(
    cnn_model, flow_model, train_loader, val_loader, num_epochs
)

torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'flow_model.pth'))
logging.info(f"Models saved to {output_dir}")

# Save losses and plot
np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN+Flow Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'cnn_flow_training_validation_loss.pdf'))