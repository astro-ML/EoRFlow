#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project imports
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
from flow import ConditionalInvertibleBlock
from data_loader import EoRFlowDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# -------------------- Config --------------------
dataset_params = {
    'max_ones_allowed': 15,
    'max_zeros_allowed': 15,
    'filter_reionization_timing': False,
    'min_redshift_index': 0,
    'max_redshift_index': 15,
    'add_noise': False,
    'n_dim': 15,
    'total_cond_dim': (15 * 10 * 10) + 15
}

pure_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/pure/test']
noise_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/noise/test']

model_base = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d'
pure_base  = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models'
model_dirs = {
    'old_pure': os.path.join(pure_base, 'pure_10_512'),
    'opt_noise': os.path.join(pure_base, 'noise_10_512'),
    'aaStar': os.path.join(model_base, 'aaStar_mod_ps2d_10_512'),
}

output_path = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/scatter_final.pdf'
N_POST_SAMPLES = 1000
USE_SIGMOID = True
BATCH_SIZE = 1
# ------------------------------------------------

def load_flow_model(model_dir):
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

def generate_stats_prediction(flow_model, condition, n_samples=N_POST_SAMPLES, use_sigmoid=USE_SIGMOID):
    """
    Returns (mean, std) of n_samples posterior draws for one condition (shape = cond_dim).
    """
    n_dim = dataset_params['n_dim']
    c = condition.unsqueeze(0).repeat(n_samples, 1).to(device)
    z = torch.randn(n_samples, n_dim, device=device)
    with torch.no_grad():
        x_samples, _ = flow_model.flow(z, c=[c], rev=True)
        if use_sigmoid:
            eps = 1e-5
            x_samples = torch.sigmoid(x_samples)
            x_samples = (x_samples - eps) / (1 - 2*eps)
        x_samples = torch.clamp(x_samples, 0, 1)
        mean = x_samples.mean(dim=0).cpu().numpy()
        std  = x_samples.std(dim=0).cpu().numpy()
    return mean, std

def predict_all(flow_model, dataset, n_samples=N_POST_SAMPLES, use_sigmoid=USE_SIGMOID):
    """
    Iterate the dataset and compute (true, mean_pred, std_pred).
    Shapes: (N, n_dim) each.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_true, all_mean, all_std = [], [], []
    for cond_tensor, true_hist in tqdm(loader, desc="Predicting"):
        # dataset already returns the condition vector (no flatten needed)
        cond_tensor = cond_tensor.squeeze(0)     # [cond_dim]
        true_hist   = true_hist.squeeze(0)       # [n_dim]
        mean_pred, std_pred = generate_stats_prediction(
            flow_model, cond_tensor, n_samples=n_samples, use_sigmoid=use_sigmoid
        )
        all_true.append(true_hist.cpu().numpy())
        all_mean.append(mean_pred)
        all_std.append(std_pred)
    return np.array(all_true), np.array(all_mean), np.array(all_std)

def build_dataset(data_dirs, noise_type=None):
    """
    Helper to build EoRFlowDataset with per-model flags matching your other scripts.
    """
    return EoRFlowDataset(
        data_dirs,
        mode='ps2d',
        min_redshift_index=dataset_params['min_redshift_index'],
        max_redshift_index=dataset_params['max_redshift_index'],
        max_ones_allowed=dataset_params['max_ones_allowed'],
        max_zeros_allowed=dataset_params['max_zeros_allowed'],
        filter_reionization_timing=dataset_params['filter_reionization_timing'],
        add_noise=(noise_type in ('pure', 'old_pure')),
        noise_level=0.05 if noise_type == 'old_pure' else 0.0,
        aa4_mod_noise=(noise_type == 'aa4'),
        aaStar_mod_noise=(noise_type == 'aaStar'),
        num_files=500,  
    )

def main():
    # Load datasets per model
    ds_old = build_dataset(pure_data_dir,  noise_type='old_pure')
    ds_opt = build_dataset(noise_data_dir, noise_type='opt_noise')
    ds_aas = build_dataset(pure_data_dir,  noise_type='aaStar')

    # Load models
    model_old = load_flow_model(model_dirs['old_pure'])
    model_opt = load_flow_model(model_dirs['opt_noise'])
    model_aas = load_flow_model(model_dirs['aaStar'])

    # Predict stats for each dataset/model
    logging.info("Predicting old_pure (slategray)…")
    true_old, mean_old, std_old = predict_all(model_old, ds_old)
    logging.info("Predicting opt_noise (red)…")
    true_opt, mean_opt, std_opt = predict_all(model_opt, ds_opt)
    logging.info("Predicting aaStar (blue)…")
    true_aas, mean_aas, std_aas = predict_all(model_aas, ds_aas)

    # Use redshifts from any dataset (all have same selected range)
    redshifts = ds_old.redshift_values
    n_dim = mean_old.shape[1]

    # Scatter grid
    n_cols = 3
    n_rows = (n_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(n_dim):
        ax = axes[i]

        # old_pure (slategray)
        ax.errorbar(
            true_old[:, i], mean_old[:, i],
            yerr=std_old[:, i],
            fmt='o', ms=3, alpha=0.35, color='slategray',
            label='Noiseless', capsize=2, linestyle='none', rasterized=True
        )
        # opt_noise (red)
        ax.errorbar(
            true_opt[:, i], mean_opt[:, i],
            yerr=std_opt[:, i],
            fmt='s', ms=3, alpha=0.35, color='red',
            label='AA4 opt', capsize=2, linestyle='none', rasterized=True
        )
        # aaStar (blue)
        ax.errorbar(
            true_aas[:, i], mean_aas[:, i],
            yerr=std_aas[:, i],
            fmt='*', ms=3, alpha=0.35, color='blue',
            label='AA* mod', capsize=2, linestyle='none', rasterized=True
        )

        # 1:1 line + limits
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(-0.01, 1.0)

        # legend per panel
        ax.legend(loc='upper left', fontsize=14, frameon=False)

        # titles and outer labels
        ax.set_title(f"$z={redshifts[i]:.2f}$", fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel(r'Predicted $x_{\rm HI}$', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'True $x_{\rm HI}$', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)

    # Hide any unused axes
    for j in range(n_dim, len(axes)):
        fig.delaxes(axes[j])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved scatter+1σ errorbars to '{output_path}'")

if __name__ == '__main__':
    main()