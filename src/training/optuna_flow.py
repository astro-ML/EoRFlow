import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import optuna

from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# ---------------------------
# Set random seeds for reproducibility
# ---------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ---------------------------
# Hyperparameters and settings (fixed ones)
# ---------------------------
lr = 0.001
batch_size = 16
num_epochs = 100           # Use a lower epoch count for tuning (adjust as needed)
early_stopping_patience = 50
min_delta = 0.0

# ---------------------------
# Output directory for logging (not used to save best model in each trial)
# ---------------------------
base_output_dir = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/optuna_study'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Set up logging
log_filename = os.path.join(base_output_dir, 'tune.log')
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # Overwrites the log file if it exists
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Change to DEBUG for more detailed logs if needed
)

# ---------------------------
# Set device
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# ---------------------------
# Prepare dataset and dataloaders
# ---------------------------
data_train = ['/lustre/fswork/projects/rech/ybg/uuv28wh/sim_data_train', '/lustre/fswork/projects/rech/ybg/uuv28wh/sim_data_filtered']
dataset = PowerSpectrumDataset(data_train)
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, val_size])
logging.info(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Define loss function for the flow model
# ---------------------------
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss

# ---------------------------
# Define training and validation function
# ---------------------------
def train_and_validate(model, train_loader, val_loader, optimizer, scheduler,
                       num_epochs, patience, min_delta):
    logging.info("Starting training for one trial...")
    
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.flow.train()
        epoch_train_loss = 0.0
        for ps_batch, target_batch in train_loader:
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

        model.flow.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                val_loss = flow_loss(model.flow, target_batch, ps_batch, model_params['flow']['n_dim'])
                epoch_val_loss += val_loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break
    return train_losses, val_losses

# ---------------------------
# Define the objective function for Optuna
# ---------------------------
def objective(trial: optuna.trial.Trial) -> float:
    # Suggest hyperparameters for the flow model.
    n_blocks = trial.suggest_int("n_blocks", 4, 12)
    n_nodes = trial.suggest_int("n_nodes", 64, 1024)
    
    # Set up the flow model parameters using the suggested values.
    global model_params  # so that train_and_validate can access it
    model_params = {
        'flow': {
            'n_dim': 3,         # Inferring 3 neutral fraction parameters.
            'n_blocks': n_blocks,
            'n_nodes': n_nodes,
            'cond_dims': 303,     # Condition dimension; adjust if needed.
            'load': False,
            'model_location': 'trained_model.pth',
            'dropout': 0.0,
        }
    }
    
    # Create the model.
    model = ConditionalInvertibleBlock(model_params)
    model.flow.to(device)
    
    # Create optimizer and scheduler.
    optimizer = optim.AdamW(model.flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Train and validate the model.
    train_losses, val_losses = train_and_validate(
        model, train_loader, val_loader, optimizer, scheduler,
        num_epochs=num_epochs,
        patience=early_stopping_patience,
        min_delta=min_delta
    )
    
    best_val_loss = min(val_losses)
    trial.report(best_val_loss, step=len(val_losses))
    
    # Optionally, you can prune trials if needed.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

# ---------------------------
# Run the Optuna study
# ---------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    study = optuna.create_study(direction="minimize", study_name="flow_hyperparam_tuning")
    study.optimize(objective, n_trials=100, timeout=3600*15)  # Run 20 trials or 1 hour, adjust as needed.
    
    logging.info("Best trial:")
    trial = study.best_trial
    logging.info(f"  Value: {trial.value:.6f}")
    logging.info("  Params:")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")
    
    # Optionally, save the study.
    study_path = os.path.join(base_output_dir, "optuna_study.pkl")
    optuna.study.write_study_pickle(study, study_path)