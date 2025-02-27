import os
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import optuna

# Import your CNN (fixed) and Flow
from cnn import CNN3D_tuned as CNN  
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset_global as PowerSpectrumDataset

##############################################################################
# 1) Define global paths and device
##############################################################################
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/optuna_tune'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(
    filename=os.path.join(output_dir, 'optuna_tune.log'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

##############################################################################
# 2) Load dataset once at the global level
##############################################################################
data_train = [
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise', 
    '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise_astro'
]
full_dataset = PowerSpectrumDataset(
    data_train,
    exclude_unfinished_reionization=True,
    exclude_early_reionization=False
)

train_ratio = 0.8
val_size = int(len(full_dataset) * (1 - train_ratio))
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# Precompute redshift array
redshift_values = np.array([ 5.        ,  5.51724138,  6.03448276,  6.55172414,  7.06896552,
                             7.5862069 ,  8.10344828,  8.62068966,  9.13793103,  9.65517241,
                            10.17241379, 10.68965517, 11.20689655, 11.72413793, 12.24137931,
                            12.75862069, 13.27586207, 13.79310345, 14.31034483, 14.82758621,
                            15.34482759, 15.86206897, 16.37931034, 16.89655172, 17.4137931 ,
                            17.93103448, 18.44827586, 18.96551724, 19.48275862, 20.        ])
redshifts_tensor = torch.tensor(redshift_values / 10, dtype=torch.float32).to(device)

##############################################################################
# 3) Define the objective function for Optuna
##############################################################################
def objective(trial):
    """
    This function tunes the Flow's flow_blocks, flow_nodes,
    plus training hyperparameters (lr, batch_size), as well as the depth of the CNN cnn_blocks and CNN dropout. 
    Returns validation loss.
    """

    # -----------------------------------------------------------
    # Suggest hyperparameters from Optuna
    # -----------------------------------------------------------
    flow_blocks = trial.suggest_int("flow_blocks", 4, 10)       
    flow_nodes  = trial.suggest_int("flow_nodes", 64, 512, step=64)  
    #lr       = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    lr = 0.001
    batch_sz = trial.suggest_categorical("batch_size", [8, 16, 32])

    # -- CNN hyperparameters
    num_conv_blocks = trial.suggest_int("cnn_blocks", 1, 4)
    final_out_dim = trial.suggest_int("cnn_out_dim", 3, 50)
    #dropout = trial.suggest_float("cnn_dropout", 0.0, 0.5)
    dropout = 0.3
    # build CNN with these hyperparams
    cnn_model = CNN(num_conv_blocks=num_conv_blocks, dropout=dropout, final_out_dim=final_out_dim).to(device)

    # -----------------------------------------------------------
    # Create DataLoaders using suggested batch size
    # -----------------------------------------------------------
    train_loader = DataLoader(train_subset, batch_size=batch_sz, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_sz, shuffle=False)

    # -----------------------------------------------------------
    # Create the Flow model w/ suggested n_blocks, n_nodes
    # -----------------------------------------------------------
    flow_params = {
        'flow': {
            'n_dim': 30,
            'n_blocks': flow_blocks,
            'n_nodes': flow_nodes,
            'cond_dims': final_out_dim + 30,  # Output size from CNN + redshift
            'load': False,
            'model_location': 'trained_model.pth',
            'dropout': 0.0,
        }
    }
    flow_model = ConditionalInvertibleBlock(flow_params)
    flow_model.flow.to(device)

    # -----------------------------------------------------------
    # Define the optimizer(s)
    # We'll keep it simpler and use a single optimizer for both
    # or you can define separate ones if you prefer
    # -----------------------------------------------------------
    optimizer = optim.AdamW(
        list(cnn_model.parameters()) + list(flow_model.flow.parameters()),
        lr=lr,
        weight_decay=1e-5
    )

    # -----------------------------------------------------------
    # Flow loss
    # -----------------------------------------------------------
    def flow_loss(flow, y, cond, n_dim):
        z, jac = flow(y, c=[cond])
        loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
        return loss.mean() / n_dim

    # -----------------------------------------------------------
    # Training loop (shorter than the final training, e.g. ~5 epochs)
    # to keep each trial quick
    # -----------------------------------------------------------
    max_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        # --- train ---
        cnn_model.train()
        flow_model.flow.train()

        train_loss_total = 0.0
        for ps_batch, target_batch in train_loader:
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            ps_batch = ps_batch.unsqueeze(1)  # (batch_size, 1, 30, 10, 10) for 3D CNN

            optimizer.zero_grad()

            # Forward pass
            redshift_batch = redshifts_tensor.repeat(ps_batch.size(0), 1)
            cnn_output = cnn_model(ps_batch, redshift_batch)
            condition = torch.cat([cnn_output, redshift_batch], dim=1)
            loss = flow_loss(flow_model.flow, target_batch, condition, 30)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), 1.0)
            optimizer.step()

            train_loss_total += loss.item()

        train_loss_avg = train_loss_total / len(train_loader)

        # --- validation ---
        cnn_model.eval()
        flow_model.flow.eval()

        val_loss_total = 0.0
        with torch.no_grad():
            for ps_batch, target_batch in val_loader:
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                ps_batch = ps_batch.unsqueeze(1)

                redshift_batch = redshifts_tensor.repeat(ps_batch.size(0), 1)
                cnn_output = cnn_model(ps_batch, redshift_batch)
                condition = torch.cat([cnn_output, redshift_batch], dim=1)
                val_loss = flow_loss(flow_model.flow, target_batch, condition, 30)
                val_loss_total += val_loss.item()

        val_loss_avg = val_loss_total / len(val_loader)

        # Use the final epoch's validation loss to track
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

        # Optional: trial.report can enable early pruning if desired
        trial.report(val_loss_avg, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return the best validation loss discovered in these epochs
    return best_val_loss

##############################################################################
# 4) Launch the Optuna study
##############################################################################
def main():
    study_name = "flow_hyperparam_optimization"
    storage = None  # or e.g. "sqlite:///optuna_storage.db" for persistent

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",  # we want to minimize val_loss
        load_if_exists=False,
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    # You can set n_trials to however many you want
    n_trials = 100

    # We optimize the objective function for n_trials
    study.optimize(objective, n_trials=n_trials)

    # After finishing, print best result
    logging.info("Study statistics:")
    logging.info(f"  Number of finished trials: {len(study.trials)}")
    logging.info("Best trial:")
    best_trial = study.best_trial
    logging.info(f"  Value (best val_loss): {best_trial.value}")
    logging.info("  Params:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # Optionally, you can load the best params from `study.best_params` and
    # do a final train with more epochs.

if __name__ == "__main__":
    main()
