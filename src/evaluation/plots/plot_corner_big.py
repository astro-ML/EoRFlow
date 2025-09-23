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
from matplotlib.ticker import FuncFormatter
from getdist import plots, MCSamples
import scienceplots
plt.style.use('science')

# adjust paths for imports
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# --- Configuration ---
fid_mid = '0_simrun_3783.npz'
fid_late = 'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz'
fid_early = 'run3249.npz'
fid_mid_2 = 'run2322.npz'
specific_file = fid_mid
pure_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/pure/fiducial']
opt_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/noise/fiducial']  
output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/corner_datasize'
os.makedirs(output_dir, exist_ok=True)

# Model directories 
model_base = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d'
pure_base = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models'
model_dirs = {
    'old_pure': os.path.join(pure_base, 'pure_10_512'),
    'opt_noise': os.path.join(pure_base, 'noise_10_512'),
    'aaStar': os.path.join(model_base, 'oldDL/aaStar_mod_ps2d_10_512'),
}


# Dataloader settings
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cpu')
logging.info(f"Using device: {device}")

def load_model(model_dir):
    """Load flow model from directory."""
    path = os.path.join(model_dir, 'best_flow_model.pth')
    cfg = {
        'flow': {
            'n_dim': dataset_params['n_dim'],
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': dataset_params['total_cond_dim'],
            'load': False,
            'model_location': path,
            'dropout': 0.0
        }
    }
    model = ConditionalInvertibleBlock(cfg)
    state = torch.load(path, map_location='cpu')
    model.flow.load_state_dict(state)
    model.flow.to(device)
    model.flow.eval()
    return model

def evaluate_model_on_file(flow_model, data_dir, file_key, noise_type=None):
    """Perform inference on one file with specified noise modification."""
    ds = PowerSpectrumDataset(
        data_dir,
        mode='ps2d',
        min_redshift_index=dataset_params['min_redshift_index'],
        max_redshift_index=dataset_params['max_redshift_index'],
        max_ones_allowed=dataset_params['max_ones_allowed'],
        max_zeros_allowed=dataset_params['max_zeros_allowed'],
        filter_reionization_timing=dataset_params['filter_reionization_timing'],
        add_noise=(noise_type in ('pure', 'old_pure')),
        noise_level=0.05 if noise_type == 'old_pure' else 0.0,
        aa4_mod_noise=(noise_type == 'aa4'),
        aaStar_mod_noise=(noise_type == 'aaStar')
    )
    idx = next(i for i, p in enumerate(ds.files) if os.path.basename(p) == file_key)
    cond_tensor, true_hist = ds[idx]
    cond = cond_tensor.to(device)
    true_hist = true_hist.to(device)

    # sample
    z = torch.randn(100000, dataset_params['n_dim'], device=device)
    cond_b = cond.unsqueeze(0).repeat(z.size(0), 1)
    x, _ = flow_model.flow(z, c=[cond_b], rev=True)

    # apply sigmoid rescaling to [0,1]
    eps = 1e-5
    x = torch.sigmoid(x)
    x = (x - eps) / (1 - 2 * eps)
    x = torch.clamp(x, 0, 1)

    samples = x.cpu().detach().numpy()
    redshifts = ds.redshift_values[:dataset_params['n_dim']]
    return redshifts, true_hist.cpu().numpy(), samples

def custom_xhi_tick_formatter(value, pos):
    if 0 < abs(value) < 0.01:
        return f'{value:.0e}'
    else:
        return f'{value:.2f}'

def plot_combined_corner_three(samples_pure, samples_opt, samples_mod,
                               redshifts, true_hist, labels,
                               out_file, n_select=15,
                               neutral_color='slategray',
                               pure_alpha=0.25, fg_alpha=0.6):
    """
    Draw a 15D corner plot with a low-alpha neutral background (old_pure),
    plus red (opt_noise) and blue (aaStar) overlays.
    """
    dim = samples_pure.shape[1]
    idxs = [int(i) for i in np.linspace(0, dim - 1, n_select)]
    sel_pure   = samples_pure[:, idxs]
    sel_opt  = samples_opt[:, idxs]
    sel_mod = samples_mod[:, idxs]
    true_sel = true_hist[idxs]

    names = [f'${redshifts[i]:.1f}$' for i in idxs]
    ranges = {}
    for i, _ in enumerate(names):
        all_vals = np.concatenate([sel_pure[:, i], sel_opt[:, i], sel_mod[:, i]])
        span = all_vals.max() - all_vals.min()
        margin = 0.05 * (span if span > 0 else 1.0)
        lo = max(0.0, all_vals.min() - margin)
        hi = min(1.0, all_vals.max() + margin)
        ranges[names[i]] = (lo, hi)

    g_pure   = MCSamples(samples=sel_pure,   names=names, labels=names, ranges=ranges)
    g_opt  = MCSamples(samples=sel_opt,  names=names, labels=names, ranges=ranges)
    g_mod = MCSamples(samples=sel_mod, names=names, labels=names, ranges=ranges)
    for s in (g_pure, g_opt, g_mod):
        s.updateSettings({"fine_bins_2D":200, "fine_bins_1D":200,
                          "smooth_scale_2D":0.9, "smooth_scale_1D":0.9})

    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.axes_fontsize = 19
    g.settings.axes_labelsize = 20
    g.settings.line_labels = False
    g.settings.legend_fontsize = 20

    # Foreground overlays
    g.settings.alpha_filled_add = fg_alpha
    g.triangle_plot([g_mod, g_pure, g_opt], filled=True, contour_colors=['blue', 'slategray', 'red'])

    # Truth markers + formatting
    fmt = FuncFormatter(custom_xhi_tick_formatter)
    for i in range(n_select):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if i == j:
                ax.axvline(true_sel[i], color='black', ls='--', lw=2)
            else:
                ax.plot(true_sel[j], true_sel[i], 'k.', ms=8)
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
            if xmin < 1e-7: ax.set_xlim(left=1e-7, right=max(xmax, 1e-5))
            if ymin < 1e-7: ax.set_ylim(bottom=1e-7, top=max(ymax, 1e-5))

    # Legend
    patch_pure   = mpatches.Patch(color='slategray', label=f'{labels[0]}', alpha=pure_alpha)
    patch_opt  = mpatches.Patch(color='red',        label=f'{labels[1]}',     alpha=fg_alpha)
    patch_mod = mpatches.Patch(color='blue',       label=f'{labels[2]}',     alpha=fg_alpha)
    true_line = mlines.Line2D([], [], color='black', marker='.', ls='--', lw=2,
                              markersize=8, label='True')
    fig = g.fig
    fig.legend(handles=[patch_pure, patch_opt, patch_mod, true_line], loc='upper right',
               bbox_to_anchor=(0.98, 0.99), frameon=False, ncol=2,
               fontsize=g.settings.legend_fontsize)

    fig.text(0.042, 0.017, r"$z=$", transform=fig.transFigure,
             ha='left', va='bottom', fontsize=19,
             bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'))
    fig.text(0.014, 0.052, r"$x_\mathrm{HI}=$", transform=fig.transFigure,
             ha='left', va='bottom', fontsize=19,
             bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'))

    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    plt.close(fig)

# --- Main workflow ---
models = {label: load_model(dir_) for label, dir_ in model_dirs.items()}

ds = PowerSpectrumDataset(
    pure_data_dir,
    min_redshift_index=dataset_params['min_redshift_index'],
    max_redshift_index=dataset_params['max_redshift_index'],
    max_ones_allowed=dataset_params['max_ones_allowed'],
    max_zeros_allowed=dataset_params['max_zeros_allowed'],
    filter_reionization_timing=dataset_params['filter_reionization_timing'],
    add_noise=dataset_params['add_noise']
)
if specific_file:
    files = [specific_file]
else:
    idxs = np.random.choice(len(ds), size=5, replace=False)
    files = [os.path.basename(ds.files[i]) for i in idxs]

for fname in files:
    logging.info(f"Processing {fname}")
    base, _ = os.path.splitext(fname)

    rz = true_hist = None
    samples = {}
    for label, model in models.items():
        data_dirs = opt_data_dir if label == 'opt_noise' else pure_data_dir
        rz_tmp, true_tmp, samp = evaluate_model_on_file(
            model, data_dirs, fname, noise_type=label)
        if rz is None:
            rz, true_hist = rz_tmp, true_tmp
        samples[label] = samp

    out_file = os.path.join(output_dir, f"corner_{base}.pdf")
    plot_combined_corner_three(
        samples_pure=samples['old_pure'],
        samples_opt=samples['opt_noise'],
        samples_mod=samples['aaStar'],
        redshifts=rz,
        true_hist=true_hist,
        labels=['Noiseless', 'AA4 opt', 'AA* mod'],
        out_file=out_file,
        neutral_color='black',  
        pure_alpha=0.8,
        fg_alpha=0.8
    )
    logging.info(f"Saved {out_file}")