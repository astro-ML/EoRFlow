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
from cnn import CNN3D_exp as CNN  # Ensure CNN includes FiLM layers
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global as PowerSpectrumDataset

# Set hyperparameters
lr_cnn = 1e-3       # Learning rate for CNN pretraining
lr_flow = 1e-3      # Learning rate for Flow training
lr_joint = 1e-4     # Learning rate for joint training
batch_size = 16
num_epochs_cnn = 100
num_epochs_flow = 200
num_epochs_joint = 300

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pretrained_exp_2'
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
data_train = [
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise_astro']
#data_train = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10']
full_dataset = PowerSpectrumDataset(data_train, exclude_unfinished_reionization=True, exclude_early_reionization=False)

redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
        7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
       10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
       12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
       15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
       17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])
redshifts = torch.tensor(redshift_values / 10, dtype=torch.float32).to(device)  # Normalize if needed

# Split dataset into train and validation
train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Initialize CNN model
cnn_model = CNN().to(device)

# --------------------------------------------------------
# Phase 1: Pretrain the CNN on xH values (Regression Task)
# --------------------------------------------------------

# Define loss function and optimizer for CNN pretraining
criterion_cnn = nn.MSELoss()
optimizer_cnn = optim.AdamW(cnn_model.parameters(), lr=lr_cnn, weight_decay=1e-5)

def pretrain_cnn(cnn_model, train_loader, val_loader, optimizer, num_epochs):
    logging.info("Starting CNN pretraining...")
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        cnn_model.train()
        for ps_batch, xH_batch in train_loader:
            ps_batch, xH_batch = ps_batch.to(device), xH_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
            ps_batch = ps_batch.to(device)
            optimizer.zero_grad()

            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # Add redshifts batch

            # Forward through CNN
            outputs = cnn_model(ps_batch, redshift_batch)
            loss = criterion_cnn(outputs, xH_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * ps_batch.size(0)

        avg_train_loss = epoch_train_loss / len(train_subset)
        train_losses.append(avg_train_loss)

        # Validation
        cnn_model.eval()
        with torch.no_grad():
            for ps_batch, xH_batch in val_loader:
                ps_batch, xH_batch = ps_batch.to(device), xH_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
                ps_batch = ps_batch.to(device)
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # Add redshifts batch

                outputs = cnn_model(ps_batch, redshift_batch)
                val_loss = criterion_cnn(outputs, xH_batch)

                epoch_val_loss += val_loss.item() * ps_batch.size(0)

        avg_val_loss = epoch_val_loss / len(val_subset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early Stopping and Saving Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'best_cnn_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

        logging.info(
            f"Pretrain CNN Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
        )

    # Load the best model before returning
    cnn_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_cnn_model.pth')))

    return train_losses, val_losses

# Pretrain the CNN
cnn_pretrain_train_losses, cnn_pretrain_val_losses = pretrain_cnn(
    cnn_model, train_loader, val_loader, optimizer_cnn, num_epochs_cnn
)

logging.info("CNN pretraining complete.")

# --------------------------------------------------------
# Phase 2: Train the Flow Model using Pretrained CNN Outputs
# --------------------------------------------------------

# Load the best pretrained CNN model
cnn_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_cnn_model.pth')))
cnn_model.eval()  # Set CNN to evaluation mode

# Initialize Flow model
model_params = {
    'flow': {
        'n_dim': 30,       # Inferring 30 xH values
        'n_blocks': 6,
        'n_nodes': 256,
        'cond_dims': 60,   # CNN output size + 30 redshifts
        'load': False,
        'model_location': 'trained_model.pth',
        'dropout': 0.3,
    }
}
flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# Define optimizer for Flow model
optimizer_flow = optim.AdamW(flow_model.flow.parameters(), lr=lr_flow, weight_decay=1e-5)

def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss

def train_flow(flow_model, cnn_model, train_loader, val_loader, optimizer, num_epochs):
    logging.info("Starting Flow model training...")
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        flow_model.flow.train()
        cnn_model.eval()  # CNN remains fixed during Flow training
        for ps_batch, xH_batch in train_loader:
            ps_batch, xH_batch = ps_batch.to(device), xH_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
            optimizer.zero_grad()

            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # Add redshifts batch

            with torch.no_grad():
                # Get conditioning input from CNN
                cond = cnn_model(ps_batch, redshift_batch)  # Shape: [batch_size, 30]
                cond = torch.cat([cond, redshift_batch], dim=1)  # Shape: [batch_size, 60]

            # Forward through Flow model
            loss = flow_loss(
                flow=flow_model.flow,
                y=xH_batch,
                cond=cond,
                n_dim=model_params['flow']['n_dim']
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * ps_batch.size(0)

        avg_train_loss = epoch_train_loss / len(train_subset)
        train_losses.append(avg_train_loss)

        # Validation
        flow_model.flow.eval()
        with torch.no_grad():
            for ps_batch, xH_batch in val_loader:
                ps_batch, xH_batch = ps_batch.to(device), xH_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # Add redshifts batch

                cond = cnn_model(ps_batch, redshift_batch)
                cond = torch.cat([cond, redshift_batch], dim=1)  # Shape: [batch_size, 60]
                val_loss = flow_loss(
                    flow=flow_model.flow,
                    y=xH_batch,
                    cond=cond,
                    n_dim=model_params['flow']['n_dim']
                )

                epoch_val_loss += val_loss.item() * ps_batch.size(0)

        avg_val_loss = epoch_val_loss / len(val_subset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early Stopping and Saving Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best Flow model
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'best_flow_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

        logging.info(
            f"Flow Training Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
        )

    # Load the best model before returning
    flow_model.flow.load_state_dict(torch.load(os.path.join(output_dir, 'best_flow_model.pth')))

    return train_losses, val_losses

# Train the Flow model
flow_train_losses, flow_val_losses = train_flow(
    flow_model, cnn_model, train_loader, val_loader, optimizer_flow, num_epochs_flow
)

logging.info("Flow model training complete.")

# --------------------------------------------------------
# Phase 3: Jointly Train the CNN and Flow Model
# --------------------------------------------------------

# Load the best pretrained CNN and Flow models
cnn_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_cnn_model.pth')))
flow_model.flow.load_state_dict(torch.load(os.path.join(output_dir, 'best_flow_model.pth')))
cnn_model.train()
flow_model.flow.train()

# Define optimizer for joint training
optimizer_joint = optim.AdamW(
    list(cnn_model.parameters()) + list(flow_model.flow.parameters()),
    lr=lr_joint,
    weight_decay=1e-5
)

def train_joint(cnn_model, flow_model, train_loader, val_loader, optimizer, num_epochs):
    logging.info("Starting joint CNN+Flow training...")
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        cnn_model.train()
        flow_model.flow.train()
        for ps_batch, xH_batch in train_loader:
            ps_batch, xH_batch = ps_batch.to(device), xH_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN
            optimizer.zero_grad()

            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # Add redshifts batch

            # Forward through CNN
            cond = cnn_model(ps_batch, redshift_batch)  # Shape: [batch_size, 30]
            cond = torch.cat([cond, redshift_batch], dim=1)  # Shape: [batch_size, 60]

            # Forward through Flow model
            loss = flow_loss(
                flow=flow_model.flow,
                y=xH_batch,
                cond=cond,
                n_dim=model_params['flow']['n_dim']
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cnn_model.parameters()) + list(flow_model.flow.parameters()), max_norm=1.0
            )
            optimizer.step()

            epoch_train_loss += loss.item() * ps_batch.size(0)

        avg_train_loss = epoch_train_loss / len(train_subset)
        train_losses.append(avg_train_loss)

        # Validation
        cnn_model.eval()
        flow_model.flow.eval()
        with torch.no_grad():
            for ps_batch, xH_batch in val_loader:
                ps_batch, xH_batch = ps_batch.to(device), xH_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1) # add dimension for 3D CNN

                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)  # Add redshifts batch

                cond = cnn_model(ps_batch, redshift_batch)
                cond = torch.cat([cond, redshift_batch], dim=1)  # Shape: [batch_size, 60]

                val_loss = flow_loss(
                    flow=flow_model.flow,
                    y=xH_batch,
                    cond=cond,
                    n_dim=model_params['flow']['n_dim']
                )

                epoch_val_loss += val_loss.item() * ps_batch.size(0)

        avg_val_loss = epoch_val_loss / len(val_subset)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early Stopping and Saving Best Models
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best CNN and Flow models
            torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'finetuned_cnn_model.pth'))
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'finetuned_flow_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

        logging.info(
            f"Joint Training Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
        )

    # Load the best models before returning
    cnn_model.load_state_dict(torch.load(os.path.join(output_dir, 'finetuned_cnn_model.pth')))
    flow_model.flow.load_state_dict(torch.load(os.path.join(output_dir, 'finetuned_flow_model.pth')))

    return train_losses, val_losses

# Jointly train the CNN and Flow model
joint_train_losses, joint_val_losses = train_joint(
    cnn_model, flow_model, train_loader, val_loader, optimizer_joint, num_epochs_joint
)

logging.info("Joint training complete.")

# --------------------------------------------------------
# Save All Losses and Plot Together
# --------------------------------------------------------

# Save all losses
np.save(os.path.join(output_dir, 'cnn_pretrain_train_losses.npy'), np.array(cnn_pretrain_train_losses))
np.save(os.path.join(output_dir, 'cnn_pretrain_val_losses.npy'), np.array(cnn_pretrain_val_losses))
np.save(os.path.join(output_dir, 'flow_train_losses.npy'), np.array(flow_train_losses))
np.save(os.path.join(output_dir, 'flow_val_losses.npy'), np.array(flow_val_losses))
np.save(os.path.join(output_dir, 'joint_train_losses.npy'), np.array(joint_train_losses))
np.save(os.path.join(output_dir, 'joint_val_losses.npy'), np.array(joint_val_losses))

# Plot all training losses
plt.figure(figsize=(12, 8))
epochs_cnn = range(1, len(cnn_pretrain_train_losses) + 1)
epochs_flow = range(len(cnn_pretrain_train_losses) + 1, len(cnn_pretrain_train_losses) + len(flow_train_losses) + 1)
epochs_joint = range(epochs_flow.stop, epochs_flow.stop + len(joint_train_losses))

plt.plot(epochs_flow, flow_train_losses, label='Flow Training Train Loss')
plt.plot(epochs_flow, flow_val_losses, label='Flow Training Val Loss')
plt.plot(epochs_joint, joint_train_losses, label='Joint Training Train Loss')
plt.plot(epochs_joint, joint_val_losses, label='Joint Training Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over All Phases')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'all_training_losses.pdf'))
plt.close()

logging.info("Training complete. All models and losses saved.")
