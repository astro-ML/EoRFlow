import os
import sys
sys.path.append('/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/models')
sys.path.append('/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/data_tools')
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset
from getdist import plots, MCSamples

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────

# Paths to your two trained models
#NOISE_MODEL_PATH = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/noise_augmented/Pk_window_10_512_std5_Gaussian0.01/best_model.pth'
NOISE_MODEL_PATH = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/pure/Pk_10_512_Gaussian0.05/best_model.pth'
PURE_MODEL_PATH = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/pure/Pk_10_512_Gaussian0.05/best_model.pth'
#PURE_MODEL_PATH  = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/pure/Pk_10_512_Gaussian0.05/best_model.pth'
#NOISE_MODEL_PATH  = '/lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/output/pure/Pk_10_512_Gaussian0.05/best_model.pth'

# PS1 & PS2 challenge files
DATASETS = {
    "PS1": {
        "noisy": './PS1/pure',
        "pure" : './PS1/pure'
    },
    "PS2": {
        "noisy": './PS2/pure',
        "pure" : './PS2/pure'
    }
}

# how many posterior draws per snapshot
SAMPLE_SIZE = 1000

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# whether to undo a logit transform after sampling
USE_SIGMOID = True
EPS = 1e-5

# optional temperature scaling (1 = no scaling)
TEMPERATURE = 1.0

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_flow(path):
    """Instantiate and load a saved flow model."""
    params = {
        'flow': {
            'n_dim': 3,
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': 303,
            'load': False,
            'model_location': '',
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    block = ConditionalInvertibleBlock(params)
    flow = block.flow.to(DEVICE)
    flow.load_state_dict(torch.load(path, map_location=DEVICE))
    flow.eval()
    return flow

def sample_posterior(flow, file_path, is_pure):
    """
    Uses PowerSpectrumDataset to load the .npz file and
    draws SAMPLE_SIZE posterior samples per spectrum.
    Returns an array of shape (N_spectra * SAMPLE_SIZE, 3).
    """
    ds = PowerSpectrumDataset(
        [file_path],
        add_noise=False,
        augment_noise=False,
        add_gaussian=True, #is_pure,
        gaussian_std=0.05 if is_pure else 0.07,#0.02,
        std_strength=1.0,
        k_scale=False,#not is_pure,
        convert_dimensionless=False,
        logit=False
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    all_samples = []
    for cond, _ in loader:
        cond = cond.to(DEVICE)                     # (1, cond_dims)
        cond_rep = cond.repeat(SAMPLE_SIZE, 1)      # (SAMPLE_SIZE, cond_dims)
        z = torch.randn(SAMPLE_SIZE, 3, device=DEVICE) * math.sqrt(TEMPERATURE)
        x, _ = flow(z, c=[cond_rep], rev=True)
        if USE_SIGMOID:
            x = torch.sigmoid(x)
            x = (x - EPS) / (1 - 2 * EPS)
        all_samples.append(x.cpu().detach().numpy())
    return np.concatenate(all_samples, axis=0)

# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__=="__main__":
    # load both flows
    flow_noise = load_flow(NOISE_MODEL_PATH)
    flow_pure  = load_flow(PURE_MODEL_PATH)

    for name, paths in DATASETS.items():
        print(f"Generating contours for {name}…")
        # draw posteriors
        samp_pure  = sample_posterior(flow_pure,  paths['pure'],  is_pure=True)
        samp_noise = sample_posterior(flow_noise, paths['noisy'], is_pure=False)
        

        mc_noise = MCSamples(
            samples=samp_noise,
            names=['xH1','xH2','xH3'],
            labels=['xH1','xH2','xH3']
        )
        mc_pure  = MCSamples(
            samples=samp_pure,
            names=['xH1','xH2','xH3'],
            labels=['xH1','xH2','xH3']
        )

        # overlay triangle plot
        g = plots.get_subplot_plotter()
        g.settings.figure_legend_frame = False
        fig = g.triangle_plot(
            [mc_pure, mc_noise],
            filled=True,
            legend_labels=['pure','pure plus small noise']
        )
        plt.suptitle(name, fontsize=14)
        fig = plt.gcf()
        fig.savefig(f'./corners_final/{name}_pure0.05+0.02noise.pdf')
  