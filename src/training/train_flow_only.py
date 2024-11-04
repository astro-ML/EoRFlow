import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import logging
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset


lr = 0.001
batch_size = 16
num_epochs = 150

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/flow_only'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
log_filename = os.path.join(output_dir, 'train.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # Overwrites the log file if it exists
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # You can change to DEBUG for more detailed logs
)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize dataset (list of folders)
data_train = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10', 
            '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2']   # 2DPS
#data_train = ['./1D_data/train']
train_dataset = PowerSpectrumDataset(data_train)

# Define the split ratios
train_ratio = 0.8  
val_ratio = 0.2  

# Calculate the sizes for the training and validation sets
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the dataset
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Check the sizes of the datasets
logging.info(f"Training set size: {len(train_subset)}")
logging.info(f"Validation set size: {len(val_subset)}")

# Initialize the model
model_params = {
    'flow': {
        'n_dim': 3,  # Inferring 3 xH values
        'n_blocks': 6, 
        'n_nodes': 256,  # 256
        'cond_dims': 300,  # Condition is the flattened 2DPS of size (3 * 10 * 10)
        'load': False,  # Load a pre-trained model
        'model_location': 'trained_model.pth',  # Location of the pre-trained model
        'dropout': 0.0,  # Dropout probability
    }
}
model = ConditionalInvertibleBlock(model_params)
model.flow.to(device)  # Move model to the GPU if available

# Set up optimizer and loss function
optimizer = optim.AdamW(model.flow.parameters(), lr=lr)  # Using AdamW optimizer

# Define the custom loss function
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) -  jac
    loss =   loss.mean() / n_dim
    return loss

def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=50):
    logging.info("Starting training...")
    model.flow.train()  # Set the model to training mode

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
            # Zero the gradients
            optimizer.zero_grad()
            # Compute loss
            loss = flow_loss(flow=model.flow, y=target_batch, cond=ps_batch, n_dim=model_params['flow']['n_dim'])
            # Backpropagation and optimization step
            loss.backward()
            # Clip the gradients to avoid exploding gradients
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


        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    logging.info("Training completed.")
    return train_losses, val_losses




# Save the model's state dictionary after training
def save_model(model, filepath):
    torch.save(model.flow.state_dict(), filepath)

# Run the training and validation
train_losses, val_losses = train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=num_epochs)

# Save the model, losses, and plot
save_model(model, os.path.join(output_dir, 'trained_model.pth'))
logging.info(f"Model saved to {os.path.join(output_dir, 'trained_model.pth')}")

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

# Save the plot as a PDF
plt.savefig(os.path.join(output_dir, 'training_validation_loss.pdf'))
logging.info(f"Plot saved to {os.path.join(output_dir, 'training_validation_loss.pdf')}")
