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
from cnn import CNN2D
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_CNN

lr = 0.001
batch_size = 16
num_epochs = 50

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/CNN_Flow_redshift'
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
data_train = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10', '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_2']
train_dataset = PowerSpectrumDataset_CNN(data_train)

# add redshift as condition
z1, z2, z3 = 6.54, 7.19, 7.96
redshifts = torch.tensor([z1/10, z2/10, z3/10], dtype=torch.float32).to(device)

# Split dataset into train and validation
train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Initialize CNN model and Flow model
input_size = (3, 10, 10)
cnn_model = CNN2D().to(device)

model_params = {
    'flow': {
        'n_dim': 3,  # Inferring 3 xH values
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': 10,  # Use CNN and redshift as condition (7 + 3)
        'load': False,
        'model_location': 'trained_model.pth',
        'dropout': 0.0,
    }
}
flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# Set up optimizer
optimizer = optim.AdamW(list(cnn_model.parameters()) + list(flow_model.flow.parameters()), lr=lr)

# Define flow loss function
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss

def train_and_validate(cnn_model, flow_model, train_loader, val_loader, optimizer, num_epochs):
    logging.info("Starting CNN+Flow training...")
    cnn_model.train()
    flow_model.flow.train()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        cnn_model.train()
        flow_model.flow.train()
        for ps_batch, target_batch in train_loader:
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()

            # Forward through CNN
            cnn_output = cnn_model(ps_batch)

            # Repeat redshift values to match the batch size
            redshift_batch = redshifts.repeat(cnn_output.size(0), 1)  # shape: [batch_size, 3]
            # Concatenate redshift values with CNN output
            condition = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, 10]
    
            # Forward through Flow using CNN output as condition
            loss = flow_loss(flow=flow_model.flow, y=target_batch, cond=condition, n_dim=model_params['flow']['n_dim'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        cnn_model.eval()
        flow_model.flow.eval()
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)

                # Forward through CNN
                cnn_output = cnn_model(ps_batch)

                # Repeat redshift values to match the batch size
                redshift_batch = redshifts.repeat(cnn_output.size(0), 1)  # shape: [batch_size, 3]
                # Concatenate redshift values with CNN output
                condition = torch.cat([cnn_output, redshift_batch], dim=1)  # shape: [batch_size, 10]

                # Forward through Flow
                val_loss = flow_loss(flow=flow_model.flow, y=target_batch, cond=condition, n_dim=model_params['flow']['n_dim'])
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    return train_losses, val_losses

# Train and save both models
train_losses, val_losses = train_and_validate(cnn_model, flow_model, train_loader, val_loader, optimizer, num_epochs)

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

