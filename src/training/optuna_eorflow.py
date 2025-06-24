import os
import sys
# Update your paths if necessary
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
# Import the flow model and dataset
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset


# hyperparameter optimization for a Conditional Invertible Flow model using Optuna.


# Configuration parameters
lr = 0.001
batch_size = 16
num_epochs = 50  # Reduced for hyperparameter search
n_trials = 50  # Number of trials for the hyperparameter search
min_redshift_index = 0
max_redshift_index = 15
redshift_dim = max_redshift_index - min_redshift_index 

# Define output directory
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/optuna_study/noise'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up logging
log_filename = os.path.join(output_dir, 'optuna_study.log')
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
#data_train=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/train'] 
# noise
data_train=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/train'] 

train_dataset = PowerSpectrumDataset(data_train, max_ones_allowed=15, max_zeros_allowed=15, 
filter_reionization_timing=False, min_redshift_index=min_redshift_index, max_redshift_index=max_redshift_index, logit=True)

# Split dataset into train and validation
train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size

# Use a fixed seed for reproducibility in the dataset split
generator = torch.Generator().manual_seed(42)
train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

# Calculate input dimensions
ps_dim = redshift_dim * 10 * 10  # Flattened power spectra
total_cond_dim = ps_dim + redshift_dim

# Define the custom loss function
def flow_loss(flow, y, cond, n_dim):
    z, jac = flow(y, c=[cond])
    loss = 0.5 * torch.sum(z ** 2, dim=1) - jac
    loss = loss.mean() / n_dim
    return loss

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_blocks = trial.suggest_int('n_blocks', 4, 20)
    n_nodes = trial.suggest_int('n_nodes', 64, 2048, log=True)  # Log scale for n_nodes
    dropout = 0.0
    weight_decay = 1e-4
    
    # Use the global batch_size 
    global batch_size
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Initialize Flow model with suggested hyperparameters
    model_params = {
        'flow': {
            'n_dim': redshift_dim,  # Inferring 11 xH values
            'n_blocks': n_blocks,
            'n_nodes': n_nodes,
            'cond_dims': total_cond_dim,  # Flattened PS + redshift dim
            'load': False,
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    
    flow_model = ConditionalInvertibleBlock(model_params)
    flow_model.flow.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(flow_model.flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_patience = 10
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training
        flow_model.flow.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            ps_batch, target_batch = batch
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            condition = ps_batch
            
            optimizer.zero_grad()
            
            # Forward through Flow using the unified loss function
            loss = flow_loss(
                flow=flow_model.flow,
                y=target_batch,
                cond=condition,
                n_dim=model_params['flow']['n_dim']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        flow_model.flow.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                ps_batch, target_batch = batch
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                condition = ps_batch
                
                # Forward through Flow
                val_loss = flow_loss(
                    flow=flow_model.flow,
                    y=target_batch,
                    cond=condition,
                    n_dim=model_params['flow']['n_dim']
                )
                
                epoch_val_loss += val_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Report to Optuna
        trial.report(avg_val_loss, epoch)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save intermediate best model for this trial
            model_path = os.path.join(output_dir, f'trial_{trial.number}_best_model.pth')
            torch.save(flow_model.flow.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
        
        # Pruning (early stopping for this trial)
        if trial.should_prune() or epochs_without_improvement >= early_stopping_patience:
            logging.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()
    
    # Log the trial results
    logging.info(f"Trial {trial.number} completed with best validation loss: {best_val_loss}")
    logging.info(f"Parameters: n_blocks={n_blocks}, n_nodes={n_nodes}, dropout={dropout}, weight_decay={weight_decay}")
    
    return best_val_loss

# Create and run the study
def run_optuna_study():
    # Create a new study
    study_name = "flow_hyperparameter_optimization"
    storage_name = os.path.join(output_dir, "optuna_study.db")
    storage = f"sqlite:///{storage_name}"
    
    sampler = TPESampler(seed=42)  # For reproducibility
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Log study start
    logging.info(f"Starting Optuna study with {n_trials} trials")
    logging.info(f"Optimizing n_blocks (6-12) and n_nodes (128-1024)")
    
    # Run the optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Log best parameters and score
    best_params = study.best_params
    best_value = study.best_value
    
    logging.info("Study completed successfully!")
    logging.info(f"Best validation loss: {best_value}")
    logging.info(f"Best hyperparameters: {best_params}")
    
    # Save the study results
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_history.pdf"))
    
    # Plot parameter importances
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "param_importances.pdf"))
    
    # Plot parameter contour plots
    plt.figure(figsize=(12, 8))
    try:
        optuna.visualization.matplotlib.plot_contour(study, params=["n_blocks", "n_nodes"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "contour_plot.pdf"))
    except:
        logging.warning("Could not create contour plot, might need more trials.")
    
    # Train final model with best parameters
    train_final_model(best_params)

# Function to train the final model with the best hyperparameters
def train_final_model(best_params):
    # Define directory for final model
    final_model_dir = os.path.join(output_dir, "final_model")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    # Use the global batch_size
    global batch_size
    
    # Create data loaders with full epochs
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with best parameters
    model_params = {
        'flow': {
            'n_dim': redshift_dim,
            'n_blocks': best_params['n_blocks'],
            'n_nodes': best_params['n_nodes'],
            'cond_dims': total_cond_dim,
            'load': False,
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    
    flow_model = ConditionalInvertibleBlock(model_params)
    flow_model.flow.to(device)
    
    # Set up optimizer with best weight decay
    optimizer = optim.AdamW(
        flow_model.flow.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Train for full number of epochs
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    epochs_without_improvement = 0
    
    logging.info(f"Training final model with best parameters: {best_params}")
    
    for epoch in range(num_epochs * 2):  # Double the epochs for final training
        # Training
        flow_model.flow.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            ps_batch, target_batch = batch
            ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
            condition = ps_batch

            optimizer.zero_grad()
            
            # Forward through Flow
            loss = flow_loss(
                flow=flow_model.flow,
                y=target_batch,
                cond=condition,
                n_dim=model_params['flow']['n_dim']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.flow.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        flow_model.flow.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                ps_batch, target_batch = batch
                ps_batch, target_batch = ps_batch.to(device), target_batch.to(device)
                condition = ps_batch
                
                # Forward through Flow
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
            f"Final Model - Epoch [{epoch+1}/{num_epochs*2}], "
            f"Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
        )
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(
                flow_model.flow.state_dict(), 
                os.path.join(final_model_dir, 'best_flow_model.pth')
            )
            logging.info(
                f"Saved new best model at epoch {epoch+1} with validation loss {avg_val_loss:.6f}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info("Early stopping triggered for final model.")
                break
    
    # Save final model regardless of performance
    torch.save(
        flow_model.flow.state_dict(), 
        os.path.join(final_model_dir, 'final_flow_model.pth')
    )
    
    # Save and plot losses
    np.save(os.path.join(final_model_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(final_model_dir, 'val_losses.npy'), np.array(val_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Final Model Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(final_model_dir, 'final_model_training_validation_loss.pdf'))
    
    # Log completion of final model training
    logging.info(f"Final model training completed with best validation loss: {best_val_loss}")
    logging.info(f"Final model saved to {final_model_dir}")

if __name__ == "__main__":
    run_optuna_study()