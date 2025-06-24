import os
import sys
# Update your paths if necessary
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import numpy as np
import logging
# Import the flow model and dataset
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# train EoRFlow

# Configuration parameters
lr = 0.001
batch_size = 16
num_epochs = 1000
weight_decay = 1e-6
min_redshift_index = 0
max_redshift_index = 15
redshift_dim = max_redshift_index - min_redshift_index 

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/pure_10_512'
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
data_train=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/train'] 
# noise
#data_train=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/train'] 

train_dataset = PowerSpectrumDataset(data_train, max_ones_allowed=15, max_zeros_allowed=15, 
filter_reionization_timing=False, min_redshift_index=min_redshift_index, max_redshift_index=max_redshift_index, 
logit=True, add_noise=True)

# Split dataset into train and validation
train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Calculate new input dimensions
ps_dim = redshift_dim * 10 * 10  # Flattened power spectra
total_cond_dim = ps_dim + redshift_dim

# Initialize Flow model with new conditioning dimensions
model_params = {
    'flow': {
        'n_dim': redshift_dim,  # Inferring 11 xH values
        'n_blocks': 10,
        'n_nodes': 512,
        'cond_dims': total_cond_dim,  # Flattened PS + redshift dim
        'load': False,
        'model_location': 'trained_model.pth',
    }
}
flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# Define the loss function
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss


def train_and_validate(flow_model, train_loader, val_loader, num_epochs):
    logging.info("Starting EoRFlow training...")
    
    flow_model.flow.train()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 20
    epochs_without_improvement = 0

    # Initialize optimizer and scheduler
    flow_optimizer = optim.AdamW(flow_model.flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=flow_optimizer, mode='min', factor=0.5, patience=10
    )

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        flow_model.flow.train()
        
        for batch in train_loader:
            ps_batch, target_batch = batch
            ps_batch = ps_batch.to(device)
            target_batch = target_batch.to(device)
            condition = ps_batch
            flow_optimizer.zero_grad()
            
            # Forward through Flow using loss function
            loss = flow_loss(
                flow=flow_model.flow,
                y=target_batch,
                cond=condition,
                n_dim=model_params['flow']['n_dim']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            flow_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        flow_model.flow.eval()
        with torch.no_grad():
            for batch in val_loader:
                ps_batch, target_batch = batch
                ps_batch = ps_batch.to(device)
                target_batch = target_batch.to(device)
                condition = ps_batch

                # Forward through Flow using the unified loss function
                val_loss = flow_loss(
                    flow=flow_model.flow,
                    y=target_batch,
                    cond=condition,
                    n_dim=model_params['flow']['n_dim']
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
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'best_flow_model.pth'))
            logging.info(f"Saved new best model at epoch {epoch+1} with validation loss {avg_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered.")
                break

    return train_losses, val_losses


# Train and save the model
train_losses, val_losses = train_and_validate(
    flow_model, train_loader, val_loader, num_epochs
)

# Save the final model
torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'flow_model.pth'))
logging.info(f"Model saved to {output_dir}")

# Save losses and plot
np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Direct Flow Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'direct_flow_training_validation_loss.pdf'))