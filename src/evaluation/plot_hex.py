#!/usr/bin/env python
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

# Make sure your modules can be imported
import sys
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

plt.style.use('science')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Output directory for plots.
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/2dhist'
os.makedirs(output_dir, exist_ok=True)

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
    state = torch.load(model_path, map_location='cpu')
    block.flow.load_state_dict(state)
    block.flow.to(device)
    block.flow.eval()
    return block

def generate_mean_prediction(flow_model, condition, n_samples=1000, use_sigmoid=False):
    n_dim = 15
    c = condition.unsqueeze(0).repeat(n_samples, 1).to(device)
    z = torch.randn(n_samples, n_dim, device=device)
    x_samples, _ = flow_model.flow(z, c=[c], rev=True)
    if use_sigmoid:
        eps = 1e-5
        x_samples = torch.sigmoid(x_samples)
        x_samples = (x_samples - eps) / (1 - 2*eps)
    return x_samples.mean(dim=0)

def predict_all(flow_model, dataset, n_samples=1000, use_sigmoid=False):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    true_hist = []
    pred_hist = []
    for ps, true in tqdm(loader, desc="Predicting"):
        ps = ps.squeeze(0)
        true = true.squeeze(0)
        cond = ps.flatten()
        with torch.no_grad():
            pred = generate_mean_prediction(flow_model, cond, n_samples, use_sigmoid)
        true_hist.append(true.cpu().numpy())
        pred_hist.append(pred.cpu().numpy())
    return np.vstack(true_hist), np.vstack(pred_hist)

def main():
    # Dataset & model params
    common_params = {
        'max_ones_allowed': 15,
        'max_zeros_allowed': 15,
        'filter_reionization_timing': False,
        'min_redshift_index': 0,
        'max_redshift_index': 15,
        'n_dim': 15,
        'total_cond_dim': 15*10*10 + 15
    }
    
    pure_params = dict(common_params, data_dir=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test'], add_noise=True)
    noise_params = dict(common_params, data_dir=['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/test'], add_noise=False)
    
    pure_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-4_extraNoise'
    noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit'
    
    # Load datasets
    pure_ds = PowerSpectrumDataset(
        pure_params['data_dir'],
        max_ones_allowed=pure_params['max_ones_allowed'],
        max_zeros_allowed=pure_params['max_zeros_allowed'],
        filter_reionization_timing=pure_params['filter_reionization_timing'],
        min_redshift_index=pure_params['min_redshift_index'],
        max_redshift_index=pure_params['max_redshift_index'],
        add_noise=pure_params['add_noise']
    )
    noise_ds = PowerSpectrumDataset(
        noise_params['data_dir'],
        max_ones_allowed=noise_params['max_ones_allowed'],
        max_zeros_allowed=noise_params['max_zeros_allowed'],
        filter_reionization_timing=noise_params['filter_reionization_timing'],
        min_redshift_index=noise_params['min_redshift_index'],
        max_redshift_index=noise_params['max_redshift_index'],
        add_noise=noise_params['add_noise']
    )
    
    # Load models
    logging.info("Loading pure model...")
    pure_model = load_flow_model(pure_model_dir, pure_params)
    logging.info("Loading noise model...")
    noise_model = load_flow_model(noise_model_dir, noise_params)
    
    # Predict
    logging.info("Predicting on pure dataset...")
    true_pure, pred_pure = predict_all(pure_model, pure_ds, n_samples=1000, use_sigmoid=True)
    logging.info("Predicting on noise dataset...")
    true_noise, pred_noise = predict_all(noise_model, noise_ds, n_samples=1000, use_sigmoid=True)
    
    def make_grid(true_arr, pred_arr, redshifts, model_name, outname):
        # true_arr, pred_arr: shape (N_samples, 15)
        fig, axes = plt.subplots(3, 5,
                                sharex=True, sharey=True,
                                figsize=(15, 9))
        axes = axes.flatten()

        mesh = None
        for idx, ax in enumerate(axes):
            tv = true_arr[:, idx]
            pv = pred_arr[:, idx]
            h = ax.hist2d(
                tv, pv,
                bins=80,
                norm=mcolors.LogNorm(),
                cmap='viridis'
            )
            # capture the QuadMesh for the colorbar
            if mesh is None:
                mesh = h[3]

            mn = min(tv.min(), pv.min())
            mx = max(tv.max(), pv.max())
            ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
            ax.set_title(f"$z={redshifts[idx]:.2f}$", fontsize=20)

        # only label the bottom row
        for ax in axes[10:]:
            ax.set_xlabel(r"True $x_\mathrm{HI}$", fontsize=18)
            ax.set_xlim(-0.01, 1)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
        # only label the left column
        for ax in axes[0::5]:
            ax.set_ylabel(r"Predicted $x_\mathrm{HI}$", fontsize=18)
            ax.set_ylim(-0.01, 1)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)

        # add one shared colorbar on the right
        cbar = fig.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            orientation='vertical',
            fraction=0.02,
            pad=0.01
        )
        cbar.set_label('counts', rotation=90, fontsize=18) 
        cbar.ax.tick_params(labelsize=16)  

        #fig.suptitle(f"2D‐histograms for {model_name}", fontsize=16)
        #fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outname, dpi=300)
        plt.close(fig)
        logging.info(f"Saved {model_name} grid → {outname}")

    # make the two figures
    make_grid(true_pure, pred_pure, pure_ds.redshift_values[:pure_params['n_dim']],
            model_name="Pure model (no noise)",
            outname="/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/grid_pure.pdf")

    make_grid(true_noise, pred_noise, noise_ds.redshift_values[:noise_params['n_dim']],
            model_name="Noise‐trained model",
            outname="/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/grid_noise.pdf")

if __name__ == '__main__':
    main()