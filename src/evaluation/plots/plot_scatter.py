import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

# Update sys.path as needed to import your modules.
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

def load_flow_model(model_dir, dataset_params):
    model_path = os.path.join(model_dir, 'best_flow_model.pth')
    cfg = {
        'flow': {
            'n_dim': dataset_params['n_dim'],
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': dataset_params['total_cond_dim'],
            'load': False,
            'model_location': model_path,
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    block = ConditionalInvertibleBlock(cfg)
    block.flow.load_state_dict(torch.load(model_path, map_location='cpu'))
    block.flow.to(device)
    block.flow.eval()
    return block

def generate_stats_prediction(flow_model, condition, n_samples=1000, use_sigmoid=False):
    """
    Returns (mean, std) of n_samples posterior draws for one condition.
    """
    n_dim = 15
    c = condition.unsqueeze(0).repeat(n_samples, 1).to(device)
    z = torch.randn(n_samples, n_dim, device=device)
    x_samples, _ = flow_model.flow(z, c=[c], rev=True)
    if use_sigmoid:
        eps = 1e-5
        x_samples = torch.sigmoid(x_samples)
        x_samples = (x_samples - eps) / (1 - 2*eps)
    # x_samples: [n_samples, n_dim]
    mean = x_samples.mean(dim=0)
    std  = x_samples.std(dim=0)
    return mean.cpu().numpy(), std.cpu().numpy()

def predict_all(flow_model, dataset, n_samples=1000, use_sigmoid=False):
    """
    Returns three arrays shaped (N, n_dim): true, mean_pred, std_pred
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_true, all_mean, all_std = [], [], []
    for ps, true_hist in tqdm(loader, desc="Predicting"):
        ps = ps.squeeze(0)
        true_hist = true_hist.squeeze(0)
        cond = ps.flatten()
        with torch.no_grad():
            mean_pred, std_pred = generate_stats_prediction(
                flow_model, cond, n_samples, use_sigmoid
            )
        all_true.append(true_hist.cpu().numpy())
        all_mean.append(mean_pred)
        all_std.append(std_pred)
    return np.array(all_true), np.array(all_mean), np.array(all_std)

def main():
    # Dataset parameters for pure and noise test datasets.
    pure_dataset_params = {
        'data_dir': ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test'],
        'max_ones_allowed': 15,
        'max_zeros_allowed': 15,
        'filter_reionization_timing': False,
        'min_redshift_index': 0,
        'max_redshift_index': 15,
        'add_noise': True,   # adjust if needed for pure data
        'n_dim': 15,
        'total_cond_dim': (15 * 10 * 10) + 15
    }
    noise_dataset_params = {
        'data_dir': ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/test'],
        'max_ones_allowed': 15,
        'max_zeros_allowed': 15,
        'filter_reionization_timing': False,
        'min_redshift_index': 0,
        'max_redshift_index': 15,
        'add_noise': False,
        'n_dim': 15,
        'total_cond_dim': (15 * 10 * 10) + 15
    }
    
    # Model directories.
    pure_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-4_extraNoise'
    noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit'
    
    # Output directory for plots.
    output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test datasets.
    pure_ds = PowerSpectrumDataset(
        pure_dataset_params['data_dir'],
        max_ones_allowed=pure_dataset_params['max_ones_allowed'],
        max_zeros_allowed=pure_dataset_params['max_zeros_allowed'],
        filter_reionization_timing=pure_dataset_params['filter_reionization_timing'],
        min_redshift_index=pure_dataset_params['min_redshift_index'],
        max_redshift_index=pure_dataset_params['max_redshift_index'],
        add_noise=pure_dataset_params['add_noise']
    )
    noise_ds = PowerSpectrumDataset(
        noise_dataset_params['data_dir'],
        max_ones_allowed=noise_dataset_params['max_ones_allowed'],
        max_zeros_allowed=noise_dataset_params['max_zeros_allowed'],
        filter_reionization_timing=noise_dataset_params['filter_reionization_timing'],
        min_redshift_index=noise_dataset_params['min_redshift_index'],
        max_redshift_index=noise_dataset_params['max_redshift_index'],
        add_noise=noise_dataset_params['add_noise']
    )
    
    # Load the models.
    logging.info("Loading pure model...")
    pure_model = load_flow_model(pure_model_dir, pure_dataset_params)
    logging.info("Loading noise model...")
    noise_model = load_flow_model(noise_model_dir, noise_dataset_params)
    
    # Predict (with stats)
    true_pure, mean_pure, std_pure   = predict_all(pure_model, pure_ds,
                                                  n_samples=1000,
                                                  use_sigmoid=True)
    true_noise, mean_noise, std_noise = predict_all(noise_model, noise_ds,
                                                  n_samples=1000,
                                                  use_sigmoid=True)

    # We assume that the true reionization histories are similar between both datasets.
    # For scatter plots we can use the true values from the pure dataset.

    
    # Use the redshift values from one of the datasets.
    redshifts = pure_ds.redshift_values
    n_dim = mean_pure.shape[1]

    # Create scatter subplots for each redshift slice.
    n_cols = 3
    n_rows = (n_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(15, 5*n_rows),
                             sharex=True, sharey=True)
    
    axes = axes.flatten()

    for i in range(n_dim):
        ax = axes[i]
        x_true_p = true_pure[:, i]
        x_true_n = true_noise[:, i]

        # pure model
        ax.errorbar(
            x_true_p,
            mean_pure[:, i],
            yerr=1*std_pure[:, i],
            fmt='o', ms=3, alpha=0.3,
            label='Posterior Noiseless', color='blue',
            capsize=2, linestyle='none', rasterized=True,
        )
        # noise model
        ax.errorbar(
            x_true_n,
            mean_noise[:, i],
            yerr=1*std_noise[:, i],
            fmt='o', ms=3, alpha=0.3,
            label='Posterior Mock', color='red',
            capsize=2, linestyle='none', rasterized=True,
        )

        # diagonal 1:1 line
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.set_xlim(-0.01,1)
        ax.set_ylim(-0.01,1)

        # legend in upper-left
        ax.legend(loc='upper left', fontsize=14, frameon=False)

        # titles and outer labels
        ax.set_title(f"$z={redshifts[i]:.2f}$", fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel(r'Predicted $x_{\rm HI}$', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
        if i >= (n_rows-1)*n_cols:
            ax.set_xlabel(r'True $x_{\rm HI}$', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)

    plt.savefig('/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/scatter_errors.pdf', dpi=300)
    plt.close(fig)
    logging.info(f"Saved scatter+2σ errorbars to '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/scatter_errors.pdf'")

if __name__ == '__main__':
    main()