#!/usr/bin/env python
import os
import sys
import numpy as np
import time
import torch
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from getdist import plots, MCSamples
import scienceplots
plt.style.use('science')

# adjust paths 
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# --- Configuration ---
fid_mid = '0_simrun_3783.npz'
fid_late = 'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz'
fid_early = 'run3249.npz'
fid_mid_2 = 'run2322.npz'
specific_file = fid_early
# for testing, eval on train data
#pure_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/train']
#noise_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/train']
pure_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test']
noise_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/test']
pure_model_dir  = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-6_bigNoise'
noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit'

use_sigmoid = True
n_samples   = 100000
num_random  = 5

output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/corner'
os.makedirs(output_dir, exist_ok=True)
cache_dir = os.path.join(output_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)

dataset_params = {
    'max_ones_allowed': 15,
    'max_zeros_allowed': 15,
    'filter_reionization_timing': False,
    'min_redshift_index': 0,
    'max_redshift_index': 15,
    'add_noise': True,
    'n_dim': 15,
    'total_cond_dim': (15 * 10 * 10) + 15
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")


def load_model(model_dir):
    """Load flow model from directory (mapping to CPU if needed)."""
    path = os.path.join(model_dir, 'best_flow_model.pth')
    cfg = {
        'flow': {
            'n_dim': dataset_params['n_dim'],
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': dataset_params['total_cond_dim'],
            'load': False,
            'model_location': path,
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    model = ConditionalInvertibleBlock(cfg)
    state = torch.load(path, map_location=torch.device('cpu'))
    model.flow.load_state_dict(state)
    model.flow.to(device)
    model.flow.eval()
    return model


def evaluate_model_on_file(flow_model, data_dir, file_key):
    """Perform inference on one file, return redshifts, true history, and posterior samples."""
    ds = PowerSpectrumDataset(
        data_dir,
        max_ones_allowed=dataset_params['max_ones_allowed'],
        max_zeros_allowed=dataset_params['max_zeros_allowed'],
        filter_reionization_timing=dataset_params['filter_reionization_timing'],
        min_redshift_index=dataset_params['min_redshift_index'],
        max_redshift_index=dataset_params['max_redshift_index'],
        add_noise=dataset_params['add_noise']
    )
    idx = next(i for i, p in enumerate(ds.files) if os.path.basename(p) == file_key)
    ps, true_hist = ds[idx]
    ps, true_hist = ps.to(device), true_hist.to(device)
    cond = ps.flatten()
    z = torch.randn(n_samples, dataset_params['n_dim'], device=device)
    cond_b = cond.unsqueeze(0).repeat(n_samples, 1)
    x, _ = flow_model.flow(z, c=[cond_b], rev=True)
    if use_sigmoid:
        eps = 1e-5
        x = torch.sigmoid(x)
        x = (x - eps) / (1 - 2 * eps)
    x = torch.clamp(x, 0, 1)
    samples = x.detach().cpu().numpy()
    redshifts = ds.redshift_values[:dataset_params['n_dim']]
    return redshifts, true_hist.cpu().numpy(), samples


def plot_combined_corner(pure_samples, noise_samples, redshifts, true_hist, out_file, n_select=15):
    """
    Plot combined pure vs. noise posterior samples on one corner plot,
    with (1) tick labels formatted to two decimal places, and
    (2) the first parameter’s label written as “z=XX.XX” while the rest are “XX.XX”.
    """
    dim = pure_samples.shape[1]
    idxs = [int(i) for i in np.linspace(0, dim - 1, n_select)]
    pure_sel = pure_samples[:, idxs]
    noise_sel = noise_samples[:, idxs]
    true_sel = true_hist[idxs]
    # Determine global min/max for each selected dimension
    ranges = {}
    for i, name in enumerate([f"${redshifts[j]:.1f}$" for j in idxs]):
        all_vals = np.concatenate([pure_sel[:, i], noise_sel[:, i]])
        margin = 0.05 * (all_vals.max() - all_vals.min())  # add some padding
        ranges[name] = (max(0, all_vals.min() - margin), min(1, all_vals.max() + margin))

    #  — Build the “names” array so that the very first entry is “z=XX.XX” and the rest are “XX.XX” —
    names = []
    for count, i in enumerate(idxs):
        if count == 0:
            # first redshift gets the “z=” prefix
            names.append(rf"${redshifts[i]:.1f}$")
        else:
            # subsequent redshifts are shown as values only (Math mode)
            names.append(f"${redshifts[i]:.1f}$")

    # Build GetDist samples
    gp = MCSamples(samples=pure_sel, names=names, labels=names, ranges=ranges)
    gn = MCSamples(samples=noise_sel, names=names, labels=names, ranges=ranges)

    gp.updateSettings({"fine_bins_2D": 200, "fine_bins_1D": 200, "smooth_scale_2D": 0.9, "smooth_scale_1D": 0.9})
    gn.updateSettings({"fine_bins_2D": 200, "fine_bins_1D": 200, "smooth_scale_2D": 0.9, "smooth_scale_1D": 0.9})

    # Create the triangle plot
    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.alpha_filled_add = 0.6
    g.settings.axes_fontsize = 21
    g.settings.axes_labelsize = 21 
    g.settings.line_labels = False
    g.settings.legend_fontsize = 20
 

    # “filled=True” draws the filled 68% region; the second call overlays the red line
    g.triangle_plot([gp, gn], filled=True,
                    contour_colors=['blue', 'red'])

    # After we’ve drawn the contours, grab every Axes and
    #   (a) overlay the “true” point
    #   (b) force its tick labels to two decimals
    for i in range(n_select):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if i == j:
                # On the diagonal, draw a vertical line at the true value
                ax.axvline(true_sel[i], color='black', ls='--', lw=2)
            else:
                # Off‐diagonal: plot the true value as a black dot
                ax.plot(true_sel[j], true_sel[i], 'k.', ms=8)

            # Force tick labels to show exactly two decimals:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Build a combined legend:
    pure_patch = mpatches.Patch(color='blue', label=r'Posterior Noiseless')
    noise_patch = mpatches.Patch(color='red', label=r'Posterior Mock')
    true_line  = mlines.Line2D([], [], color='black', marker='.', ls='--', lw=2,
                               markersize=8, label='True')
    fig = g.fig
    fig.legend(
        handles=[pure_patch, noise_patch, true_line],
        loc='upper right',
        bbox_to_anchor=(0.98, 0.99),
        fontsize=g.settings.legend_fontsize,
        frameon=False,
        ncol=2
    )

    fig.text(
        0.05, 0.017,           
        r"$z=$",
        transform=fig.transFigure,
        ha='left', va='bottom',
        fontsize=19,
        bbox=dict(facecolor='white', alpha=0.1, edgecolor='none')
    )

    fig.text(
        0.022, 0.055,           
        r"$x_\mathrm{HI}=$",
        transform=fig.transFigure,
        ha='left', va='bottom',
        fontsize=19,
        bbox=dict(facecolor='white', alpha=0.1, edgecolor='none')
    )

    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    plt.close(fig)


# --- Main workflow ---
pure_model = load_model(pure_model_dir)
noise_model = load_model(noise_model_dir)

pure_ds = PowerSpectrumDataset(
    pure_data_dir,
    max_ones_allowed=dataset_params['max_ones_allowed'],
    max_zeros_allowed=dataset_params['max_zeros_allowed'],
    filter_reionization_timing=dataset_params['filter_reionization_timing'],
    min_redshift_index=dataset_params['min_redshift_index'],
    max_redshift_index=dataset_params['max_redshift_index'],
    add_noise=dataset_params['add_noise']
)
noise_ds = PowerSpectrumDataset(
    noise_data_dir,
    max_ones_allowed=dataset_params['max_ones_allowed'],
    max_zeros_allowed=dataset_params['max_zeros_allowed'],
    filter_reionization_timing=dataset_params['filter_reionization_timing'],
    min_redshift_index=dataset_params['min_redshift_index'],
    max_redshift_index=dataset_params['max_redshift_index'],
    add_noise=dataset_params['add_noise']
)

if specific_file is not None:
    files = [specific_file]
else:
    chosen_idxs = np.random.choice(len(pure_ds), size=num_random, replace=False)
    files = [os.path.basename(pure_ds.files[i]) for i in chosen_idxs]

for fname in files:
    logging.info(f"Processing {fname}")
    base, ext = os.path.splitext(fname)

    rz_path    = os.path.join(cache_dir, f"{base}_rz.npy")
    th_path    = os.path.join(cache_dir, f"{base}_th.npy")
    pure_path  = os.path.join(cache_dir, f"{base}_pure.npy")
    noise_path = os.path.join(cache_dir, f"{base}_noise.npy")

    if all(os.path.exists(p) for p in (rz_path, th_path, pure_path, noise_path)):
        logging.info("  loading cached samples")
        rz         = np.load(rz_path)
        th         = np.load(th_path)
        pure_samp  = np.load(pure_path)
        noise_samp = np.load(noise_path)
    else:
        t0 = time.time()
        rz, th, pure_samp  = evaluate_model_on_file(pure_model, pure_data_dir, fname)
        dt = time.time() - t0
        logging.info(f"pure network eval took {dt:.1f}s")

        t0 = time.time()
        #noisy_name = fname.replace('.npz','_noisy.npz')
        noisy_name = fname
        _, _, noise_samp = evaluate_model_on_file(noise_model, noise_data_dir, noisy_name)
        dt = time.time() - t0
        logging.info(f"noise network eval took {dt:.1f}s")

        logging.info("  caching to .npy")
        np.save(rz_path,    rz)
        np.save(th_path,    th)
        np.save(pure_path,  pure_samp)
        np.save(noise_path, noise_samp)

    out_file = os.path.join(output_dir, f"full_corner_{base}_HQ.pdf")
    plot_combined_corner(pure_samp, noise_samp, rz, th, out_file)
    logging.info(f"Saved {out_file}")