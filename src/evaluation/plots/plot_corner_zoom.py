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
import matplotlib.ticker as mticker
from getdist import plots, MCSamples
import scienceplots
plt.style.use('science')

# adjust paths so that `flow` and `FilteredPowerSpectrumDataset` can be imported
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
pure_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test']
noise_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/test']
pure_model_dir  = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-6_bigNoise'
noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit'

use_sigmoid = True
n_samples   = 100000 # Increased as per user script
num_random  = 5

output_dir = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/corner'
os.makedirs(output_dir, exist_ok=True)
cache_dir = os.path.join(output_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)

# Global dataset parameters (base configuration)
dataset_params_config = {
    'max_ones_allowed': 15,
    'max_zeros_allowed': 15,
    'filter_reionization_timing': False,
    'min_redshift_index': 0,
    'max_redshift_index': 15,
    'add_noise': True, # Default, will be overridden for pure data
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
            'n_dim': dataset_params_config['n_dim'], # Use from global config
            'n_blocks': 10,
            'n_nodes': 512,
            'cond_dims': dataset_params_config['total_cond_dim'], # Use from global config
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


def evaluate_model_on_file(flow_model, data_dir, file_key, add_noise_override):
    """Perform inference on one file, return redshifts, true history, and posterior samples."""
    # Use a copy of global params and apply override
    current_params = dataset_params_config.copy()
    current_params['add_noise'] = add_noise_override

    ds = PowerSpectrumDataset(
        data_dir,
        max_ones_allowed=current_params['max_ones_allowed'],
        max_zeros_allowed=current_params['max_zeros_allowed'],
        filter_reionization_timing=current_params['filter_reionization_timing'],
        min_redshift_index=current_params['min_redshift_index'],
        max_redshift_index=current_params['max_redshift_index'],
        add_noise=current_params['add_noise'] # Use the overridden value
    )
    idx = next(i for i, p in enumerate(ds.files) if os.path.basename(p) == file_key)
    ps, true_hist = ds[idx]
    ps, true_hist = ps.to(device), true_hist.to(device)
    cond = ps.flatten()
    z = torch.randn(n_samples, current_params['n_dim'], device=device)
    cond_b = cond.unsqueeze(0).repeat(n_samples, 1)
    x, _ = flow_model.flow(z, c=[cond_b], rev=True)
    if use_sigmoid:
        eps = 1e-5
        x = torch.sigmoid(x)
        x = (x - eps) / (1 - 2 * eps)
    x = torch.clamp(x, 0, 1)
    samples = x.detach().cpu().numpy()
    redshifts = ds.redshift_values[:current_params['n_dim']]
    return redshifts, true_hist.cpu().numpy(), samples


def custom_xhi_tick_formatter(value, pos):
    """
    Formats tick labels for xHI values.
    Uses scientific notation for values < 0.001 (and > 0).
    Uses standard decimal format (0.XXX) otherwise.
    If value is 0, formats as '0'.
    """
    if abs(value) < 1e-9: # Treat very small numbers as 0 for formatting
        return '0'
    elif 0 < abs(value) < 0.001:
        return f'{value:.0e}'  # e.g., 1e-4, 5e-5
    else:
        return f'{value:.3f}' # e.g., 0.123, 0.001


def plot_combined_corner(pure_samples, noise_samples, redshifts, true_hist, out_file, n_select=4):
    dim = pure_samples.shape[1]
    original_idxs = [int(i) for i in np.linspace(0, dim - 1, n_select)]
    swapped_idxs = original_idxs[::-1]

    pure_sel = pure_samples[:, swapped_idxs]
    noise_sel = noise_samples[:, swapped_idxs]
    
    ranges = {}
    for i, name_idx in enumerate(swapped_idxs):
        all_vals = np.concatenate([pure_sel[:, i], noise_sel[:, i]])
        data_min, data_max = all_vals.min(), all_vals.max()
        span = data_max - data_min
        margin = 0.05 * span if span > 1e-9 else 0.01
        
        # Ensure range is valid for GetDist (min < max)
        r_min = max(0, data_min - margin)
        r_max = min(1, data_max + margin)
        if r_min >= r_max - 1e-7: # If range is too small or inverted
            if r_max < 1.0 - 1e-5:
                r_max = r_min + 1e-5
            else:
                r_min = r_max - 1e-5
            r_min = max(0.0, r_min)
            r_max = min(1.0, r_max)
        ranges[f'p{i}'] = (r_min, r_max)

    true_sel = true_hist[swapped_idxs]

    getdist_names = []
    plot_labels = []
    for k, original_idx_in_swapped_order in enumerate(swapped_idxs):
        getdist_names.append(f'p{k}') 
        redshift_val = redshifts[original_idx_in_swapped_order]
        plot_labels.append(rf"$x_\mathrm{{HI}}(z={redshift_val:.1f})$") # Removed extra spaces

    gp = MCSamples(samples=pure_sel, names=getdist_names, labels=plot_labels, label='Posterior Noiseless', ranges=ranges)
    gn = MCSamples(samples=noise_sel, names=getdist_names, labels=plot_labels, label='Posterior Mock', ranges=ranges)

    gp.updateSettings({"fine_bins_2D": 200, "fine_bins_1D": 200, "smooth_scale_2D": 0.9, "smooth_scale_1D": 0.9})
    gn.updateSettings({"fine_bins_2D": 200, "fine_bins_1D": 200, "smooth_scale_2D": 0.9, "smooth_scale_1D": 0.9})

    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.alpha_filled_add = 0.6
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.line_labels = False
    g.settings.legend_fontsize = 20
    
    x_label_padding = 15
    y_label_padding = 30 

    g.triangle_plot([gp, gn], filled=True, contour_colors=['blue', 'red'],
                    upper_roots=[gp, gn], upper_label_right=True, upper=True,
                    legend_labels=['Posterior Noiseless', 'Posterior Mock'])
    
    xhi_formatter = mticker.FuncFormatter(custom_xhi_tick_formatter)

    for i_row in range(n_select):
        for j_col in range(n_select):
            ax = g.subplots[i_row, j_col]
            if ax is None: continue

            if i_row == j_col:
                ax.axvline(true_sel[i_row], color='black', ls='--', lw=2)
            else:
                ax.plot(true_sel[j_col], true_sel[i_row], 'k.', ms=8)

            ax.xaxis.set_major_formatter(xhi_formatter)
            ax.yaxis.set_major_formatter(xhi_formatter)

            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=2, prune=None)) # nbins=2 for 3 ticks
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, prune=None)) # nbins=2 for 3 ticks

            ax.tick_params(axis='both', which='major', labelsize=g.settings.axes_fontsize)
            ax.xaxis.labelpad = x_label_padding
            ax.yaxis.labelpad = y_label_padding

            # Parameter labels and tick positions
            if i_row == 0: 
                ax.set_xlabel(plot_labels[j_col], fontsize=g.settings.axes_labelsize)
                ax.xaxis.set_label_position('top'); ax.xaxis.label.set_visible(True)
                ax.tick_params(axis='x', labeltop=True, top=True, labelbottom=False, bottom=False)
                if j_col == 0 : ax.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False); ax.set_ylabel('') 
            elif i_row == n_select - 1: 
                ax.set_xlabel(plot_labels[j_col], fontsize=g.settings.axes_labelsize)
                ax.tick_params(axis='x', labelbottom=True, bottom=True, labeltop=False, top=False)
            elif i_row == j_col: 
                ax.tick_params(axis='x', labelbottom=True, bottom=True, labeltop=False, top=False); ax.set_xlabel('') 
                ax.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False); ax.set_ylabel('') 
            else: 
                ax.tick_params(axis='x', labelbottom=False, bottom=False, labeltop=False, top=False); ax.set_xlabel('')
                ax.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False); ax.set_ylabel('')

            if j_col == 0 and i_row > 0 : 
                ax.set_ylabel(plot_labels[i_row], fontsize=g.settings.axes_labelsize)
                ax.tick_params(axis='y', labelleft=True, left=True, labelright=False, right=False)
            elif j_col == n_select - 1 and i_row < j_col: 
                ax.set_ylabel(plot_labels[i_row], fontsize=g.settings.axes_labelsize)
                ax.yaxis.set_label_position('right'); ax.yaxis.label.set_visible(True)
                ax.tick_params(axis='y', labelright=True, right=True, labelleft=False, left=False)

    if hasattr(g, 'subplots_diag_twinx'):
        for k_diag in range(n_select): 
            if k_diag < len(g.subplots_diag_twinx) and g.subplots_diag_twinx[k_diag]:
                twin_ax = g.subplots_diag_twinx[k_diag]
                twin_ax.set_ylabel(plot_labels[k_diag], fontsize=g.settings.axes_labelsize)
                twin_ax.yaxis.label.set_visible(True)
                twin_ax.yaxis.labelpad = y_label_padding
                twin_ax.tick_params(axis='y', labelsize=g.settings.axes_fontsize,
                                    labelright=True, right=True, labelleft=False, left=False)
                # DO NOT apply xhi_formatter to probability density axis. Let it use default.
                # twin_ax.yaxis.set_major_formatter(xhi_formatter) # REMOVED/COMMENTED OUT

    fig = g.fig
    try:
        # fig.tight_layout(pad=1.5) # Sometimes problematic, try adjusting subplot parameters first
        fig.subplots_adjust(hspace=0.05, wspace=0.05) # Small space
        fig.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5) # Then try to fit        
    except (ValueError, RuntimeError) as e:
        logging.warning(f"tight_layout or subplots_adjust failed: {e}. Check plot appearance carefully.")

    logging.info("Attempting final x-axis inversion for all subplots...")
    for i_row_final in range(n_select): 
        for j_col_final in range(n_select): 
            final_ax = g.subplots[i_row_final, j_col_final]
            if final_ax is not None:
                current_xlim = final_ax.get_xlim()
                if current_xlim[0] < current_xlim[1]: # Only invert if not already inverted
                    final_ax.set_xlim(current_xlim[1], current_xlim[0])

    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    plt.close(fig)


# --- Main workflow ---
pure_model = load_model(pure_model_dir)
noise_model = load_model(noise_model_dir)

# Datasets are loaded inside evaluate_model_on_file or for file listing
if specific_file is not None:
    files = [specific_file]
else:
    # Temporarily create dataset instance to list files
    temp_ds_params = dataset_params_config.copy()
    temp_ds_params['add_noise'] = False # For listing, add_noise doesn't matter
    temp_ds = PowerSpectrumDataset(
        pure_data_dir,
        max_ones_allowed=temp_ds_params['max_ones_allowed'],
        max_zeros_allowed=temp_ds_params['max_zeros_allowed'],
        filter_reionization_timing=temp_ds_params['filter_reionization_timing'],
        min_redshift_index=temp_ds_params['min_redshift_index'],
        max_redshift_index=temp_ds_params['max_redshift_index'],
        add_noise=temp_ds_params['add_noise']
    )
    if not temp_ds.files:
        logging.error("Pure dataset (for file listing) has no files. Exiting.")
        sys.exit(1)
    chosen_idxs = np.random.choice(len(temp_ds.files), size=min(num_random, len(temp_ds.files)), replace=False)
    files = [os.path.basename(temp_ds.files[i]) for i in chosen_idxs]
    del temp_ds


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
        logging.info("  Generating samples as cache not found or incomplete.")
        
        # Pure samples evaluation
        time_pure_start = time.time()
        rz, th, pure_samp = evaluate_model_on_file(pure_model, pure_data_dir, fname, add_noise_override=False)
        dt_pure = time.time() - time_pure_start
        logging.info(f"pure network eval took {dt_pure:.1f}s")

        # Noise samples evaluation
        # Determine noisy file name
        noisy_file_to_find = fname 
        actual_noise_file_path_found = any(os.path.exists(os.path.join(root_dir, noisy_file_to_find)) for root_dir in noise_data_dir)
        
        if not actual_noise_file_path_found:
            alternative_noisy_name = fname.replace('.npz','_noisy.npz')
            logging.warning(f"Noisy file {noisy_file_to_find} not found in {noise_data_dir}. Trying {alternative_noisy_name}")
            if any(os.path.exists(os.path.join(root_dir, alternative_noisy_name)) for root_dir in noise_data_dir):
                noisy_file_to_find = alternative_noisy_name
                actual_noise_file_path_found = True
            else:
                logging.error(f"Alternative noisy file {alternative_noisy_name} also not found. Skipping noise evaluation for {fname}.")
                # If pure samples were generated, cache them at least
                if 'pure_samp' in locals():
                    np.save(rz_path, rz)
                    np.save(th_path, th)
                    np.save(pure_path, pure_samp)
                    logging.info("  Cached pure samples only.")
                continue 
        
        time_noise_start = time.time()
        # For the call below, ensure add_noise_override=True for noisy data
        _, _, noise_samp = evaluate_model_on_file(noise_model, noise_data_dir, noisy_file_to_find, add_noise_override=True)
        dt_noise = time.time() - time_noise_start
        logging.info(f"noise network eval took {dt_noise:.1f}s")

        logging.info("  caching to .npy")
        np.save(rz_path,    rz)
        np.save(th_path,    th)
        np.save(pure_path,  pure_samp)
        np.save(noise_path, noise_samp)

    out_file = os.path.join(output_dir, f"zoom_corner_{base}.pdf")
    plot_combined_corner(pure_samp, noise_samp, rz, th, out_file) 
    logging.info(f"Saved {out_file}")

logging.info("All processing finished.")