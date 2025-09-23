#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from getdist import plots, MCSamples
import scienceplots
plt.style.use('science')

# --- imports ---
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
from flow import ConditionalInvertibleBlock
from data_loader import EoRFlowDataset, PowerSpectrumDataset

# --- model dirs ---
fid_mid = '0_simrun_3783.npz'
fid_late = 'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz'
fid_early = 'run3249.npz'
fid_mid_2 = 'run2322.npz'
specific_file = fid_early

pure_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/pure/test']
opt_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/noise/test']
output_dir    = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/corner_final'
os.makedirs(output_dir, exist_ok=True)

model_base = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d'
pure_base  = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models'
model_dirs = {
    'old_pure': os.path.join(pure_base, 'pure_10_512'),
    'opt_noise': os.path.join(pure_base, 'noise_10_512'),
    'aaStar': os.path.join(model_base, 'oldDL/aaStar_mod_ps2d_10_512_oldDL'),
}

# --- Dataset / model config  ---
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

N_SAMPLES   = 100_000
DEVICE      = torch.device('cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f'Using device: {DEVICE}')

# ---- Choose which 4 slices to zoom on ----
# Option A: explicit indices 
select_idxs = None  # e.g., [0, 5, 10, 14]
# Option B: if None, pick 4 evenly spaced

def load_model(model_dir):
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
    model.flow.to(DEVICE)
    model.flow.eval()
    return model

def evaluate_model_on_file(flow_model, data_dirs, file_key, noise_type=None):
    ds = PowerSpectrumDataset(
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
        aaStar_mod_noise=(noise_type == 'aaStar')
    )
    idx = next(i for i, p in enumerate(ds.files) if os.path.basename(p) == file_key)
    cond_tensor, true_hist = ds[idx]
    cond = cond_tensor.to(DEVICE)
    true_hist = true_hist.to(DEVICE)

    z = torch.randn(N_SAMPLES, dataset_params['n_dim'], device=DEVICE)
    cond_b = cond.unsqueeze(0).repeat(z.size(0), 1)
    x, _ = flow_model.flow(z, c=[cond_b], rev=True)

    # rescale to [0,1]
    eps = 1e-5
    x = torch.sigmoid(x)
    x = (x - eps) / (1 - 2 * eps)
    x = torch.clamp(x, 0, 1)

    samples = x.detach().cpu().numpy()
    redshifts = ds.redshift_values[:dataset_params['n_dim']]
    return redshifts, true_hist.cpu().numpy(), samples

def custom_xhi_tick_formatter(value, pos):
    if abs(value) < 1e-6:
        return '0'
    elif 0 < abs(value) < 1e-2:
        return f'{value:.0e}'
    else:
        return f'{value:.2f}'

def plot_zoom_corner_three(samples_oldpure, samples_opt, samples_aastar,
                           redshifts, true_hist, out_file, idxs4=None):
    """
    4x4 corner with BOTH triangles + diagonal.
    Labels/ticks on TOP and RIGHT only, with consistent font sizes and padding.
    """
    # ---------- styling knobs ----------
    tick_fontsize  = 18
    label_fontsize = 18
    tick_pad       = 8     # space between ticks and their labels
    x_label_pad    = 18    # space between top-axis label text and axes
    y_label_pad    = 34    # space between right-axis label text and axes
    hspace, wspace = 0.06, 0.06
    top_margin     = 0.90  # leave room for top labels
    right_margin   = 0.97  # leave room for right labels
    # -----------------------------------

    # --- choose 4 dims ---
    dim = samples_oldpure.shape[1]
    if idxs4 is None:
        idxs4 = [int(i) for i in np.linspace(0, dim - 1, 4)]
    swapped = idxs4[::-1]  # high->low z visual ordering

    sel_old  = samples_oldpure[:, swapped]
    sel_opt  = samples_opt[:, swapped]
    sel_aas  = samples_aastar[:, swapped]
    true_sel = true_hist[swapped]

    # Build ranges shared by all three
    ranges, names, labels = {}, [], []
    for k, orig_idx in enumerate(swapped):
        all_vals = np.concatenate([sel_old[:, k], sel_opt[:, k], sel_aas[:, k]])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        span = vmax - vmin
        margin = 0.05 * span if span > 1e-9 else 1e-2
        rmin = max(0.0, vmin - margin)
        rmax = min(1.0, vmax + margin)
        if rmax <= rmin + 1e-7:
            rmax = min(1.0, rmin + 1e-5)
        names.append(f"p{k}")
        labels.append(rf"$x_\mathrm{{HI}}(z={redshifts[orig_idx]:.1f})$")
        ranges[names[-1]] = (rmin, rmax)

    g_old = MCSamples(samples=sel_old, names=names, labels=labels, label='Noiseless', ranges=ranges)
    g_opt = MCSamples(samples=sel_opt, names=names, labels=labels, label='AA4 opt', ranges=ranges)
    g_aas = MCSamples(samples=sel_aas, names=names, labels=labels, label='AA* mod', ranges=ranges)
    for s in (g_old, g_opt, g_aas):
        s.updateSettings({"fine_bins_2D": 200, "fine_bins_1D": 200,
                          "smooth_scale_2D": 0.9, "smooth_scale_1D": 0.9})

    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.alpha_filled_add  = 0.8
    g.settings.axes_fontsize     = tick_fontsize   # base text on axes
    g.settings.axes_labelsize    = label_fontsize  # base label size
    g.settings.line_labels       = False
    g.settings.legend_fontsize   = label_fontsize  # no legend anyway

    # BOTH triangles (lower default + upper)
    g.triangle_plot([g_aas, g_old, g_opt],
                    filled=True,
                    contour_colors=['blue', 'slategray', 'red'],
                    upper=True,
                    upper_roots=[g_aas, g_old, g_opt],
                    upper_label_right=True)

    fmt = mticker.FuncFormatter(custom_xhi_tick_formatter)
    n = 4
    for i in range(n):
        for j in range(n):
            ax = g.subplots[i, j]
            if ax is None:
                continue

            # truth markers
            if i == j:
                ax.axvline(true_sel[i], color='black', ls='--', lw=2)
            else:
                ax.plot(true_sel[j], true_sel[i], 'k.', ms=8)

            # formatting
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

            # small-value guard
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            mindisp = 1e-7
            if xmin < mindisp: ax.set_xlim(left=mindisp, right=max(xmax, 1e-5))
            if ymin < mindisp: ax.set_ylim(bottom=mindisp, top=max(ymax, 1e-5))

            # hide bottom/left ticks
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
            ax.tick_params(axis='y', labelleft=False,   left=False)

            # show top row x-labels + ticks ON TOP with consistent size & padding
            if i == 0:
                ax.set_xlabel(labels[j], fontsize=label_fontsize)
                ax.xaxis.set_label_position('top')
                ax.xaxis.labelpad = x_label_pad
                ax.tick_params(axis='x', labeltop=True, top=True,
                               labelsize=tick_fontsize, pad=tick_pad)

            # show rightmost col y-labels + ticks ON RIGHT with consistent size & padding
            if j == n - 1:
                ax.set_ylabel(labels[i], fontsize=label_fontsize)
                ax.yaxis.set_label_position('right')
                ax.yaxis.labelpad = y_label_pad
                ax.tick_params(axis='y', labelright=True, right=True,
                               labelsize=tick_fontsize, pad=tick_pad)

            # keep tick density light
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))

    # If GetDist creates diagonal twin y-axes, make sure they don't mess with sizes/padding
    if hasattr(g, 'subplots_diag_twinx'):
        for k in range(n):
            if k < len(g.subplots_diag_twinx) and g.subplots_diag_twinx[k]:
                twin = g.subplots_diag_twinx[k]
                # put twin label on right to match our scheme
                twin.yaxis.set_label_position('right')
                twin.tick_params(axis='y', labelright=True, right=True,
                                 labelsize=tick_fontsize, pad=tick_pad)
                # don't relabel; leave as density scale

    # Optional: invert x-axes to match your previous zoom visual
    for i in range(n):
        for j in range(n):
            ax = g.subplots[i, j]
            if ax is not None:
                x0, x1 = ax.get_xlim()
                if x0 < x1:
                    ax.set_xlim(x1, x0)

    fig = g.fig
    # No legend (removed)
    # Layout: give extra room at top/right to avoid any collisions
    fig.subplots_adjust(hspace=hspace, wspace=wspace, top=top_margin, right=right_margin)
    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    plt.close(fig)
# --- Main ---
if __name__ == '__main__':
    # load models
    models = {k: load_model(v) for k, v in model_dirs.items()}

    # pick file(s)
    ds = PowerSpectrumDataset(
        pure_data_dir,
        mode='ps2d',
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
        idxs = np.random.choice(len(ds), size=10, replace=False)
        files = [os.path.basename(ds.files[i]) for i in idxs]

    for fname in files:
        logging.info(f"Processing zoom for {fname}")
        base, _ = os.path.splitext(fname)

        rz = true_hist = None
        samp = {}
        for label, model in models.items():
            data_dirs = opt_data_dir if label == 'opt_noise' else pure_data_dir
            rz_i, th_i, s_i = evaluate_model_on_file(model, data_dirs, fname, noise_type=label)
            if rz is None:
                rz, true_hist = rz_i, th_i
            samp[label] = s_i

        out_file = os.path.join(output_dir, f"{base}_zoom.pdf")
        plot_zoom_corner_three(
            samples_oldpure=samp['old_pure'],
            samples_opt=samp['opt_noise'],
            samples_aastar=samp['aaStar'],
            redshifts=rz,
            true_hist=true_hist,
            out_file=out_file,
            idxs4=select_idxs
        )
        logging.info(f"Saved {out_file}")