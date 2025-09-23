#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import matplotlib as mpl

# LaTeX rendering + serif
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# --- imports ---
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')
from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# -------- Configuration --------
fid_mid   = '0_simrun_3783.npz'
fid_late  = 'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz'
fid_early = 'run3249.npz'
fid_mid_2 = 'run2322.npz'

specific_file   = fid_mid       # or None to pick random
num_random      = 5
plot_data       = False
output_dir      = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/EoR_hist/VERYLAST'
os.makedirs(output_dir, exist_ok=True)

# Data roots
pure_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/pure/test']
noise_data_dir = ['/remote/gpu01a/pietschke/EoRFlow/data/power_spectra/noise/test']

# Models 
model_base = '/remote/gpu01a/pietschke/EoRFlow/output/ps2d'
pure_base  = '/remote/gpu01a/pietschke/EoRFlow/output/paper_models'
model_dirs = {
    'old_pure': os.path.join(pure_base, 'pure_10_512'),
    'opt_noise': os.path.join(pure_base, 'noise_10_512'),
    'aaStar': os.path.join(model_base, 'oldDL/aaStar_mod_ps2d_10_512_oldDL'),
}

# Sampling + dataset params 
use_sigmoid = True
n_samples   = 1000

dataset_params = {
    'max_ones_allowed': 15,
    'max_zeros_allowed': 15,
    'filter_reionization_timing': False,
    'min_redshift_index': 0,
    'max_redshift_index': 15,
    'add_noise': False,  
    'n_dim': 15,
    'total_cond_dim': 15*10*10 + 15
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")


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
            'dropout': 0.0,
            'sigmoid': False
        }
    }
    m = ConditionalInvertibleBlock(cfg)
    m.flow.load_state_dict(torch.load(path, map_location='cpu'))
    m.flow.to(device)
    m.flow.eval()
    return m


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
    cond = cond_tensor.to(device)
    true_hist = true_hist.to(device)

    z = torch.randn(n_samples, dataset_params['n_dim'], device=device)
    cond_b = cond.unsqueeze(0).repeat(n_samples, 1)
    x, _ = flow_model.flow(z, c=[cond_b], rev=True)

    if use_sigmoid:
        eps = 1e-5
        x = torch.sigmoid(x)
        x = (x - eps) / (1 - 2*eps)
    x = torch.clamp(x, 0, 1)

    return ds.redshift_values[:dataset_params['n_dim']], true_hist.cpu().numpy(), x.detach().cpu().numpy()


def plot_history_three(zs, stats, th, align_str, out_file, plot_data=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    # aaStar
    ma = stats['aaStar']['mean']
    sa = stats['aaStar']['std']
    ax.plot(zs, ma, '-', color='blue', lw=2, label='AA* mod')
    ax.fill_between(zs, ma-2*sa, ma+2*sa, color='blue', alpha=0.25)

    # old_pure
    mp = stats['old_pure']['mean']
    sp = stats['old_pure']['std']
    ax.plot(zs, mp, '-', color='slategray', lw=2, label='Noiseless')
    ax.fill_between(zs, mp-2*sp, mp+2*sp, color='slategray', alpha=0.3)

    # opt_noise
    mn = stats['opt_noise']['mean']
    sn = stats['opt_noise']['std']
    ax.plot(zs, mn, 'r-', lw=2, label='AA4 opt')
    ax.fill_between(zs, mn-2*sn, mn+2*sn, color='red', alpha=0.3)

    # True history
    ax.plot(zs, th, 'k.-', lw=1.5, ms=6, label='True')

    if plot_data:
        x = [5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7]
        y = [0.1, 0.17, 0.28, 0.68, 0.79, 0.86, 0.93]
        upper_err = np.array([0.05, 0.1, 0.05, 0.03, 0.03, 0.01, 0.04])
        lower_arrow = 0.1
        ax.errorbar(x, y, yerr=[np.zeros_like(upper_err), upper_err], fmt='o',
                    elinewidth=2, capsize=4, color='gold', label='_nolegend_')
        ax.errorbar(x, y, yerr=[lower_arrow*np.ones_like(y), upper_err], fmt='none',
                    uplims=True, elinewidth=2, capsize=4, color='gold')

        ax.errorbar(5.3, 3e-5, yerr=[[0.125e-5],[0.466e-5]], color='dodgerblue',
                    fmt='o', elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar([7, 8.8, 11], [0.49, 0.82, 0.9],
                    xerr=[[0.5, 0.8, 2.1], [1, 1.2, 2]],
                    yerr=[[0.23, 0.34, 0.22], [0.14, 0.12, 0.08]],
                    fmt='o', capsize=4, elinewidth=2, label='_nolegend_', color='darkcyan')

        ax.errorbar(7.68, 0.5, xerr=0.79, fmt='o', elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar(7.0, 0.59, yerr=[[0.15],[0.11]], fmt='o', elinewidth=2, capsize=4, label='_nolegend_')

        ax.errorbar(5.6, 0.19, yerr=[[0.16],[0.11]], fmt='o', color='darkgreen',
                    elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar(5.9, 0.44, yerr=0.1, uplims=True, fmt='o', color='darkgreen',
                    elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar(11.5, 0.8, yerr=0.2, fmt='o', color='darkorange',
                    elinewidth=2, capsize=4, label='_nolegend_'),
        ax.errorbar([6.15, 6.35], [0.2, 0.29], yerr=[[0.12, 0.13],[0.14, 0.14]],
                    color='fuchsia', fmt='o', elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar([5.8, 5.95, 6.05, 6.55], [0.21, 0.2, 0.21, 0.18], yerr=0.1, uplims=True,
                    color='fuchsia', fmt='o', elinewidth=2, capsize=4, label='_nolegend_')

    # Axes, limits, legend, layout 
    ax.set_xlabel(r'$z$', fontsize=45)
    ax.set_ylabel(r'$x_{\mathrm{HI}}$', fontsize=45)
    ax.set_xlim(zs.min(), zs.max())
    ax.set_ylim(-0.05, 1.0)
    ax.tick_params(labelsize=36)

    handles, labels = ax.get_legend_handles_labels()
    order = [3,1,0,2]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=35, loc='lower right') 
    #ax.legend(fontsize=35, loc='lower right')
    fig = ax.get_figure()
    fig.subplots_adjust(left=0.10, right=0.88, top=0.95, bottom=0.12)
    fig.savefig(out_file, dpi=400)
    plt.close(fig)


def main():
    # load 3 models
    models = {k: load_model(v) for k, v in model_dirs.items()}

    # list files from pure set
    ds_pure = PowerSpectrumDataset(
        pure_data_dir,
        mode='ps2d',
        min_redshift_index=dataset_params['min_redshift_index'],
        max_redshift_index=dataset_params['max_redshift_index'],
        max_ones_allowed=dataset_params['max_ones_allowed'],
        max_zeros_allowed=dataset_params['max_zeros_allowed'],
        filter_reionization_timing=dataset_params['filter_reionization_timing'],
        add_noise=dataset_params['add_noise']
    )
    all_files = [os.path.basename(p) for p in ds_pure.files]

    if specific_file:
        chosen = [specific_file]
    else:
        chosen = list(np.random.choice(all_files, size=min(num_random, len(all_files)), replace=False))

    for fname in chosen:
        logging.info(f"Processing {fname}")

        # try to extract params from pure file if present 
        try:
            fullpath = next(p for p in ds_pure.files if os.path.basename(p) == fname)
            params = np.load(fullpath).get('params', None)
            if params is not None and len(params) >= 6:
                m_wdm, Omega_m, E0, LX_exp, Tvir_exp, zeta = params[:6]
                align_vals = {
                    'Omega_m': Omega_m, 'm_wdm': m_wdm, 'E0': E0,
                    'LX_exp': LX_exp, 'Tvir_exp': Tvir_exp, 'zeta': zeta
                }
            else:
                align_vals = {}
        except StopIteration:
            align_vals = {}

        # find noisy filename match for opt_noise
        base, _ = os.path.splitext(fname)
        ds_noise = PowerSpectrumDataset(
            noise_data_dir,
            mode='ps2d',
            min_redshift_index=dataset_params['min_redshift_index'],
            max_redshift_index=dataset_params['max_redshift_index'],
            max_ones_allowed=dataset_params['max_ones_allowed'],
            max_zeros_allowed=dataset_params['max_zeros_allowed'],
            filter_reionization_timing=dataset_params['filter_reionization_timing'],
            add_noise=False
        )
        noise_match = next(
            (os.path.basename(p) for p in ds_noise.files
             if os.path.splitext(os.path.basename(p))[0] in {base, base}),
            None
        )
        if noise_match is None:
            logging.warning(f"No noise match for {fname}; skipping")
            continue

        # Evaluate three models
        zs, th, samp_old  = evaluate_model_on_file(models['old_pure'], pure_data_dir, fname,  noise_type='old_pure')
        _,  _, samp_opt   = evaluate_model_on_file(models['opt_noise'], noise_data_dir, noise_match, noise_type='opt_noise')
        _,  _, samp_aas   = evaluate_model_on_file(models['aaStar'],    pure_data_dir, fname,  noise_type='aaStar')

        stats = {
            'old_pure': {'mean': samp_old.mean(axis=0), 'std': samp_old.std(axis=0)},
            'opt_noise': {'mean': samp_opt.mean(axis=0), 'std': samp_opt.std(axis=0)},
            'aaStar': {'mean': samp_aas.mean(axis=0), 'std': samp_aas.std(axis=0)},
        }

        out = os.path.join(output_dir, f"concept_{base}.pdf")
        plot_history_three(zs, stats, th, align_vals, out, plot_data=plot_data)
        logging.info(f"Saved {out}")


if __name__ == '__main__':
    main()  