import os
import sys
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Add custom paths
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import EoRFlowDataset, PowerSpectrumDataset

# If using CNN:
from cnn import ConvNet3D

# ------------------- CONFIG -------------------
mode = 'ps2d'       # Options: 'ps2d', 'ps1d', or 'cnn'
n_blocks = 10
n_nodes = 512

# tag/output folder
out_tag = f'{n_blocks}_{n_nodes}'
output_dir = f'/remote/gpu01a/pietschke/EoRFlow/output/{mode}/aaStar_mod_{out_tag}'
os.makedirs(output_dir, exist_ok=True)

# hyperparams
lr = 1e-3
batch_size = 16
num_epochs = 1000
weight_decay = 1e-4

min_redshift_index = 0
max_redshift_index = 15
redshift_dim = max_redshift_index - min_redshift_index

# Logging
logging.basicConfig(
    filename=os.path.join(output_dir, 'train.log'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ------------------- DEVICE -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- DATA -------------------
data_train = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/pure/train']

if mode in ['ps2d', 'ps1d']:
    train_dataset = PowerSpectrumDataset(
        data_dirs=data_train,
        mode=mode,
        logit=True,
        add_noise=False,
        noise_level=1.0,  
        aa4_mod_noise=False,
        aaStar_mod_noise=True,
        min_redshift_index=min_redshift_index,
        max_redshift_index=max_redshift_index
    )
else:  # cnn, depends on data format, consistent version will come later
    image_dirs = ['/remote/gpu01a/heneka/21cmlightcones/pure_simulations']
    train_dataset = EoRFlowDataset(
        data_dirs=data_train,
        image_dirs=image_dirs,
        mode='cnn',
        logit=True,
        min_redshift_index=min_redshift_index,
        max_redshift_index=max_redshift_index
    )

# Sanity checks before DataLoader
logging.info(f"Dataset size = {len(train_dataset)}")
logging.info(f"Mode = {mode}")


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)

# ------------------- MODEL -------------------
# 1) Instantiate the CNN (if needed)
if mode == 'cnn':
    cnn_params = {'cnn': {'load': False, 'model_location': 'trained_cnn.pth'}}
    cnn_model = ConvNet3D(cnn_params, in_ch=1, N_parameter=6)
    cnn_model.to(device)
    cond_dims = 6                # must match ConvNet3D’s N_parameter
else:
    # previous PS conditioning dims
    if mode == 'ps2d':
        ps_dim = redshift_dim * 10 * 10
    else:
        ps_dim = redshift_dim * 10
    cond_dims = ps_dim + redshift_dim

model_params = {
    'flow': {
        'n_dim': redshift_dim,
        'n_blocks': n_blocks,
        'n_nodes': n_nodes,
        'cond_dims': cond_dims,
        'load': False,
        'model_location': 'trained_model.pth',
    }
}

flow_model = ConditionalInvertibleBlock(model_params)
flow_model.flow.to(device)

# ------------------- LOSS -------------------
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    return (0.5 * torch.sum(z**2, dim=1) - jac).mean() / n_dim

# ------------------- TRAIN & VALIDATE -------------------
def train_and_validate(flow_model, train_loader, val_loader, num_epochs):
    logging.info("Starting EoRFlow training...")
    optimizer = optim.AdamW(flow_model.flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    patience, no_improve = 20, 0

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs+1):
        # ——— Training ———
        logging.info(f"Epoch {epoch}/{num_epochs} — Training...")
        flow_model.flow.train()
        total_train = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if mode == 'cnn':
                imgs, y, zs = batch
                imgs, y = imgs.to(device), y.to(device)
                # add channel dim
                cond = cnn_model(imgs.unsqueeze(1))
            else:
                cond, y = batch
                cond, y = cond.to(device), y.to(device)

            loss = flow_loss(flow_model.flow, y, cond, model_params['flow']['n_dim'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        # ——— Validation ———
        flow_model.flow.eval()
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if mode == 'cnn':
                    imgs, y, zs = batch
                    imgs, y = imgs.to(device), y.to(device)
                    cond = cnn_model(imgs.unsqueeze(1))
                else:
                    cond, y = batch
                    cond, y = cond.to(device), y.to(device)

                total_val += flow_loss(flow_model.flow, y, cond, model_params['flow']['n_dim']).item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        logging.info(f"Epoch {epoch}/{num_epochs} — Train: {avg_train:.6f}, Val: {avg_val:.6f}")
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val, no_improve = avg_val, 0
            torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'best_flow_model.pth'))
            logging.info(f"  → New best model at epoch {epoch}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info("Early stopping.")
                break

    return train_losses, val_losses

# ------------------- RUN -------------------
train_losses, val_losses = train_and_validate(flow_model, train_loader, val_loader, num_epochs)

# save final model & curves
torch.save(flow_model.flow.state_dict(), os.path.join(output_dir, 'flow_model.pth'))
np.save(os.path.join(output_dir, 'train_losses.npy'), train_losses)
np.save(os.path.join(output_dir, 'val_losses.npy'), val_losses)

plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Train')
plt.plot(val_losses,   label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'EoRFlow Training Loss ({mode})')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_curve.pdf'))