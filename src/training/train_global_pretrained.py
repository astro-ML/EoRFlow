import os
import sys
import logging
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Update your paths if necessary
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

# Import your CNN, Flow and Dataset
from cnn import CNN3D_film as CNN  
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global_full as PowerSpectrumDataset

# Configuration parameters
lr = 0.001
batch_size = 16
total_joint_epochs = 1000  # Maximum epochs for joint training

# Set output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/full_EoR_pure_TomsData_pretrained_noFilter'
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
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/toms_data_pure'
]
train_dataset = PowerSpectrumDataset(data_train, exclude_unfinished_reionization=False, exclude_early_reionization=False)

# Define redshift values (and normalize if needed)
redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
                             7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
                            10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
                            12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
                            15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
                            17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])
redshifts = torch.tensor(redshift_values / 10, dtype=torch.float32).to(device)

# Split dataset into training and validation subsets
train_ratio = 0.8
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Initialize models
# Note: The CNN input size and output dimensions should match your network design.
cnn_model = CNN().to(device)

model_params = {
    'flow': {
        'n_dim': 30,  # number of parameters to infer
        'n_blocks': 7,
        'n_nodes': 384,
        'cond_dims': 30+30,  # Adjust: CNN output + redshift dimension (here redshift has dimension 20, CNN output is assumed to be 30)
        'dropout': 0.0,
        'load': False,
        'model_location': 'trained_model.pth',
    }
}
flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# ============================
# Stage 1: Pretrain CNN only
# ============================
def pretrain_cnn(cnn_model, train_loader, val_loader, num_epochs, device):
    logging.info("Starting CNN pretraining...")
    cnn_model.train()
    optimizer = optim.AdamW(cnn_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        cnn_model.train()
        epoch_train_loss = 0.0
        for ps_batch, target_batch in train_loader:
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1)  # add channel dimension for CNN

            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
            # Forward through CNN
            cnn_output = cnn_model(ps_batch, redshift_batch)
            # Pretraining objective: directly regress to target_batch
            loss = criterion(cnn_output, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        cnn_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1)
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
                cnn_output = cnn_model(ps_batch, redshift_batch)
                loss = criterion(cnn_output, target_batch)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"[Pretrain CNN] Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save best CNN model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'pretrained_cnn_model.pth'))
            logging.info(f"Saved best CNN model at epoch {epoch+1}")

    return train_losses, val_losses

# ============================
# Stage 2: Pretrain Flow only
# ============================
def pretrain_flow(cnn_model, flow_model, train_loader, val_loader, num_epochs, device):
    logging.info("Starting Flow pretraining (with CNN frozen)...")
    # Freeze CNN parameters
    for param in cnn_model.parameters():
        param.requires_grad = False

    flow_model.flow.train()
    optimizer = optim.AdamW(flow_model.flow.parameters(), lr=lr, weight_decay=1e-5)

    # Use the flow loss as before:
    def flow_loss(flow, y, cond, n_dim):
        z, jac = flow(y, c=[cond])
        loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
        return loss.mean() / n_dim

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        flow_model.flow.train()
        epoch_train_loss = 0.0
        for ps_batch, target_batch in train_loader:
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1)
            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
            # Use the pretrained CNN to produce condition
            with torch.no_grad():
                cnn_output = cnn_model(ps_batch, redshift_batch)
            condition = torch.cat([cnn_output, redshift_batch], dim=1)
            loss = flow_loss(flow_model.flow, target_batch, condition, model_params['flow']['n_dim'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        flow_model.flow.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1)
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
                cnn_output = cnn_model(ps_batch, redshift_batch)
                condition = torch.cat([cnn_output, redshift_batch], dim=1)
                loss = flow_loss(flow_model.flow, target_batch, condition, model_params['flow']['n_dim'])
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"[Pretrain Flow] Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'pretrained_flow_model.pth'))
            logging.info(f"Saved best Flow model at epoch {epoch+1}")

    # Unfreeze CNN after flow pretraining
    for param in cnn_model.parameters():
        param.requires_grad = True

    return train_losses, val_losses

# ============================
# Stage 3: Joint Training
# ============================
def joint_train(cnn_model, flow_model, train_loader, val_loader, num_epochs, device):
    logging.info("Starting joint training of CNN and Flow...")
    cnn_model.train()
    flow_model.flow.train()
    optimizer = optim.AdamW(
        list(cnn_model.parameters()) + list(flow_model.flow.parameters()),
        lr=lr,
        weight_decay=1e-5
    )

    def flow_loss(flow, y, cond, n_dim):
        z, jac = flow(y, c=[cond])
        loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
        return loss.mean() / n_dim

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 50
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        cnn_model.train()
        flow_model.flow.train()
        epoch_train_loss = 0.0
        for ps_batch, target_batch in train_loader:
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1)
            redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
            cnn_output = cnn_model(ps_batch, redshift_batch)
            condition = torch.cat([cnn_output, redshift_batch], dim=1)
            loss = flow_loss(flow_model.flow, target_batch, condition, model_params['flow']['n_dim'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        cnn_model.eval()
        flow_model.flow.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1)
                redshift_batch = redshifts.repeat(ps_batch.size(0), 1)
                cnn_output = cnn_model(ps_batch, redshift_batch)
                condition = torch.cat([cnn_output, redshift_batch], dim=1)
                loss = flow_loss(flow_model.flow, target_batch, condition, model_params['flow']['n_dim'])
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"[Joint Training] Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'best_cnn_model.pth'))
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'best_flow_model.pth'))
            logging.info(f"Saved new best joint model at epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered in joint training.")
                break

    return train_losses, val_losses

# ============================
# Run the Three Stages
# ============================
# You can adjust these epoch numbers as needed:
pretrain_cnn_epochs = 200
pretrain_flow_epochs = 200
joint_epochs = total_joint_epochs  # up to 1000 epochs (with early stopping)

# Stage 1: Pretrain CNN
logging.info("=== Stage 1: Pretraining CNN ===")
cnn_train_losses, cnn_val_losses = pretrain_cnn(cnn_model, train_loader, val_loader, pretrain_cnn_epochs, device)

# Stage 2: Pretrain Flow (with CNN frozen)
logging.info("=== Stage 2: Pretraining Flow ===")
flow_train_losses, flow_val_losses = pretrain_flow(cnn_model, flow_model, train_loader, val_loader, pretrain_flow_epochs, device)

# Stage 3: Joint Training of CNN and Flow
logging.info("=== Stage 3: Joint Training ===")
joint_train_losses, joint_val_losses = joint_train(cnn_model, flow_model, train_loader, val_loader, joint_epochs, device)

# Save final models
torch.save(cnn_model.state_dict(), os.path.join(output_dir, 'cnn_model_final.pth'))
torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'flow_model_final.pth'))
logging.info(f"Final models saved to {output_dir}")

# Save loss curves
np.save(os.path.join(output_dir, 'cnn_pretrain_train_losses.npy'), np.array(cnn_train_losses))
np.save(os.path.join(output_dir, 'cnn_pretrain_val_losses.npy'), np.array(cnn_val_losses))
np.save(os.path.join(output_dir, 'flow_pretrain_train_losses.npy'), np.array(flow_train_losses))
np.save(os.path.join(output_dir, 'flow_pretrain_val_losses.npy'), np.array(flow_val_losses))
np.save(os.path.join(output_dir, 'joint_train_losses.npy'), np.array(joint_train_losses))
np.save(os.path.join(output_dir, 'joint_val_losses.npy'), np.array(joint_val_losses))

# Plot training curves (example for joint training)
plt.figure(figsize=(10, 6))
plt.plot(joint_train_losses, label='Joint Training Loss')
plt.plot(joint_val_losses, label='Joint Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Joint Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'joint_training_validation_loss.pdf'))
plt.show()