import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scienceplots
plt.style.use('science')
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
from data_loader import FilteredPowerSpectrumDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_reionization_histories(data_dirs,
                                output_dir="./plots",
                                max_zeros_allowed=15,
                                max_ones_allowed=15,
                                alpha=0.01,
                                filter_reionization_timing=False,
                                highlight_files=None):
    """
    Plot all reionization histories, optionally highlighting a subset.

    Args:
        data_dirs (list of str): directories with the .npz files.
        output_dir (str): where to write the PDF.
        max_zeros_allowed (int): passed to dataset.
        max_ones_allowed (int): passed to dataset.
        alpha (float): line transparency for unhighlighted histories.
        filter_reionization_timing (bool): passed to dataset.
        highlight_files (list of str): basenames (e.g. 'sim_1234.npz') to highlight.
    """
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    dataset = FilteredPowerSpectrumDataset(
        data_dirs=data_dirs,
        max_zeros_allowed=max_zeros_allowed,
        max_ones_allowed=max_ones_allowed,
        filter_reionization_timing=filter_reionization_timing
    )
    files = [os.path.basename(p) for p in dataset.files]
    logging.info(f"Dataset contains {len(dataset)} histories")

    # redshifts
    redshift_values = dataset.redshift_values
    z_orig = redshift_values
    z_fine = np.linspace(z_orig.min(), z_orig.max(), 300)

    # gaussian‐PCHIP params
    sigma = 0.8

    # precompute highlight set
    highlight_set = set(highlight_files) if highlight_files else set()

    # start plotting
    plt.figure(figsize=(12, 8))
    for i in tqdm(range(len(dataset)), desc="Plotting"):
        # load one history
        _, label_tensor = dataset[i]
        xh = label_tensor.numpy()

        # smooth + monotonic interp
        xh_sm = gaussian_filter1d(xh, sigma=sigma, mode='reflect')
        xh_sm = np.clip(xh_sm, 0.0, 1.0)
        pchip = PchipInterpolator(z_orig, xh_sm)
        xh_smooth = pchip(z_fine)

        # decide style
        fname = files[i]
        if fname in highlight_set:
            col = 'green'
            lw = 2.0
            aa = 1.0
            zorder = 2
        else:
            col = 'dodgerblue'
            lw = 0.8
            aa = alpha
            zorder = 1

        # plot
        plt.plot(z_fine, xh_smooth,
                 color=col,
                 alpha=aa,
                 linewidth=lw,
                 rasterized=True,
                 zorder=zorder)

    # axes formatting
    plt.tick_params(axis='both', which='major', labelsize=26)
    plt.tick_params(axis='both', which='minor', labelsize=24)
    plt.xlabel(r'$z$', fontsize=32)
    plt.ylabel(r'$x_\mathrm{HI}$', fontsize=32)
    plt.ylim(0, 1)
    plt.xlim(z_orig.min(), z_orig.max())

    # legend for highlights
    bg_line = mlines.Line2D([], [], color='dodgerblue', lw=2, alpha=0.6, label='Simulated models')
    if highlight_set:
        hl_line = mlines.Line2D([], [], color='forestgreen', lw=2, alpha=0.6, label='Fiducial models')
    plt.legend(handles=[bg_line, hl_line], fontsize=26)

    # save
    out = os.path.join(output_dir, 'reionization_histories.pdf')
    plt.tight_layout()
    plt.savefig(out, dpi=400, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot to {out}")


# Example usage:
plot_reionization_histories(
    data_dirs=[
        '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/train',
        '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test'
    ],
    output_dir='./plots',
    max_zeros_allowed=15,
    max_ones_allowed=15,
    alpha=0.015,
    highlight_files=[
        '0_simrun_3783.npz',
        'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz',
        'run3249.npz',
        'run2322.npz',
    ]
)
