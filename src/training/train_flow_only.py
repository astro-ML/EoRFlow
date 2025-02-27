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
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparameters and settings
lr = 0.001
batch_size = 16
num_epochs = 1000

# Early stopping parameters
early_stopping_patience = 20   # Number of epochs with no improvement to wait before stopping
min_delta = 0.0                # Minimum change in the monitored quantity to qualify as an improvement

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/SKA_flow/flow_6_256_sampling'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
log_filename = os.path.join(output_dir, 'train.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # Overwrites the log file if it exists
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Change to DEBUG for more detailed logs if needed
)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize dataset (list of folders)
data_train = [
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2',
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_noise'
]
train_dataset = PowerSpectrumDataset(data_train, skip_boundary_labels=True, skip_fraction=0.9)

# Define the split ratios and split the dataset
train_ratio = 0.8  
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Log dataset sizes
logging.info(f"Training set size: {len(train_subset)}")
logging.info(f"Validation set size: {len(val_subset)}")

# Initialize the model
model_params = {
    'flow': {
        'n_dim': 3,         # Inferring 3 neutral fraction parameters (for 3 redshifts)
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': 303,    # Condition is the flattened 2DPS of size (3 * 10 * 10); double-check if 303 is intended vs 300
        'load': False,       # Whether to load a pre-trained model
        'model_location': 'trained_model.pth',
        'dropout': 0.0,
    }
}
model = ConditionalInvertibleBlock(model_params)
model.flow.to(device)  # Move model to GPU if available

# Set up optimizer (using AdamW)
optimizer = optim.Adam(model.flow.parameters(), lr=lr)

# Set up the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,      
    verbose=True
)

# Define the custom loss function
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss

def train_and_validate(model, train_loader, val_loader, optimizer, scheduler, num_epochs=50,
                       patience=10, min_delta=0.0):
    logging.info("Starting training with early stopping and LR scheduler...")
    
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        
        # --- Training ---
        model.flow.train()
        for batch_idx, (ps_batch, target_batch) in enumerate(train_loader):
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            loss = flow_loss(flow=model.flow, y=target_batch, cond=ps_batch, 
                             n_dim=model_params['flow']['n_dim'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.flow.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.flow.eval()
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                val_loss = flow_loss(model.flow, target_batch, ps_batch, model_params['flow']['n_dim'])
                epoch_val_loss += val_loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, "
                     f"Validation Loss: {avg_val_loss:.6f}")

        # --- Scheduler Step ---
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}")

        # --- Early Stopping Check ---
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model state dictionary
            best_model_state = model.flow.state_dict()
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(best_model_state, best_model_path)
            logging.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("Training completed.")
    return train_losses, val_losses

# Run the training and validation with early stopping and LR scheduler
train_losses, val_losses = train_and_validate(
    model, train_loader, val_loader, optimizer, scheduler,
    num_epochs=num_epochs,
    patience=early_stopping_patience,
    min_delta=min_delta
)

# Save the final model (from the last epoch)
def save_model(model, filepath):
    torch.save(model.flow.state_dict(), filepath)

final_model_path = os.path.join(output_dir, 'trained_model.pth')
save_model(model, final_model_path)
logging.info(f"Final model saved to {final_model_path}")

# Save the training and validation losses
np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))
logging.info("Training and validation losses saved as .npy files")

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, 'training_validation_loss.pdf')
plt.savefig(plot_path)
logging.info(f"Plot saved to {plot_path}")