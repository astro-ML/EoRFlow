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

# Turn on full LaTeX rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# adjust module paths
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/models')
sys.path.append('/remote/gpu01a/pietschke/EoRFlow/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import PowerSpectrumDataset

# --- Configuration ---
fid_mid = '0_simrun_3783.npz'
fid_late = 'lightcone_5z25CDMOMm0.316E0222.325LX40.993Tvir5.532Zeta83.402.npz'
fid_early = 'run3249.npz'
fid_mid_2 = 'run2322.npz'

specific_file   = fid_mid_2 #fid_mid_2 #   # or None to pick random
num_random      = 20
plot_data       = False
output_dir      = '/remote/gpu01a/pietschke/EoRFlow/output/paper_plots/EoR_hist/final_four'
os.makedirs(output_dir, exist_ok=True)

pure_data_dir   = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/pure/test']
noise_data_dir  = ['/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/noise/test']
pure_model_dir  = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/pure_z12_10_512_-6_bigNoise'
noise_model_dir = '/remote/gpu01a/pietschke/EoRFlow/output/EoR_flow_logit/10_512_noise_wd-4_logit'

use_sigmoid = True
n_samples   = 1000

dataset_params = {
    'max_ones_allowed': 15,
    'max_zeros_allowed': 15,
    'filter_reionization_timing': False,
    'min_redshift_index': 0,
    'max_redshift_index': 15,
    'add_noise': True,
    'n_dim': 15,
    'total_cond_dim': 15*10*10 + 15
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
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


def evaluate_model_on_file(flow_model, data_dir, file_key):
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
        x = (x - eps) / (1 - 2*eps)
    x = torch.clamp(x, 0, 1)
    return ds.redshift_values[:dataset_params['n_dim']], true_hist.cpu().numpy(), x.cpu().detach().numpy()


def plot_history(zs, mp, sp, mn, sn, th, align_str, out_file, plot_data=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    #for y in (0, 0.5):
    #    ax.axhline(y, color='k', ls=':', lw=1)

    ax.plot(zs, mp, 'b-', lw=2, label='Noiseless')
    ax.fill_between(zs, mp-2*sp, mp+2*sp, color='blue', alpha=0.3)

    ax.plot(zs, mn, 'r--', lw=2, label='Mock')
    ax.fill_between(zs, mn-2*sn, mn+2*sn, color='red', alpha=0.3)

    ax.plot(zs, th, 'k.-', lw=1.5, ms=6, label='True')

    if plot_data:
        x = [5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7]
        y = [0.1, 0.17, 0.28, 0.68, 0.79, 0.86, 0.93]
        upper_err = np.array([0.05, 0.1, 0.05, 0.03, 0.03, 0.01, 0.04])
        lower_arrow = 0.1
    
        # everything from Qin, Mesinger et al 2024 (Percent level timing)
        # Jin, Yang, Fan et al 2022 ((Nearly) Model independent) Dark Pixels
        ax.errorbar(
            x, y,
            yerr=[np.zeros_like(upper_err), upper_err],  # zero‐lower, real upper
            fmt='o',
            elinewidth=2,
            capsize=4,
            color='gold',
            label='_nolegend_'
        )
        ax.errorbar(
            x, y,
            yerr=[lower_arrow * np.ones_like(y), upper_err],  # real lower, zero upper
            fmt='none',      # no extra marker
            uplims=True,     # draw a downward arrow
            elinewidth=2,
            capsize=4,
            color='gold')

        # Bosman, Davies, Becker et al 2022 (Hydrogen reionization ends by z=5.3...) Lyalpha forest
        ax.errorbar(5.3, 3e-5, yerr=[[0.125e-5],[0.466e-5]], color='dodgerblue', fmt='o', elinewidth=2, capsize=4, label='_nolegend_')

        # Tang, Stark, Topping et al 2024 (JWST/NIRSpec) Lyalpha EWs
        ax.errorbar([7, 8.8, 11], [0.49, 0.82, 0.9], xerr=[[0.5, 0.8, 2.1], [1, 1.2, 2]], yerr=[[0.23, 0.34, 0.22], [0.14, 0.12, 0.08]], fmt='o',
                   capsize=4, elinewidth=2, label='_nolegend_', color='darkcyan')
        

        # Planck 2020 (Cosmic Microwave Background)
        ax.errorbar(7.68, 0.5, xerr=0.79, fmt='o', elinewidth=2, capsize=4, label='_nolegend_')
        # Mason 2018 LAE
        ax.errorbar(7.0, 0.59, yerr=[[0.15],[0.11]], fmt='o', elinewidth=2, capsize=4, label='_nolegend_')

        # Spina 2024 Lyalpha forest DW
        ax.errorbar(5.6, 0.19, yerr=[[0.16],[0.11]], fmt='o', color='darkgreen',
                    elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar(5.9, 0.44, yerr=0.1, uplims=True, fmt='o', color='darkgreen',
                    elinewidth=2, capsize=4, label='_nolegend_')
        ax.errorbar(11.5, 0.8, yerr=0.2, fmt='o', color='darkorange', elinewidth=2, capsize=4, label='_nolegend_'),
        # Greig 2024 DW
        ax.errorbar([6.15, 6.35], [0.2, 0.29], yerr=[[0.12, 0.13],[0.14, 0.14]], color='fuchsia', fmt='o', elinewidth=2,
                   capsize=4, label='_nolegend_')
        ax.errorbar([5.8, 5.95, 6.05, 6.55], [0.21, 0.2, 0.21, 0.18], yerr=0.1, uplims=True, color='fuchsia', fmt='o', elinewidth=2,
                   capsize=4, label='_nolegend_')
        

    # draw aligned block (no leading whitespace!)
    ax.text(
        0.05, 0.95,
        "Intermediate",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=35,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    """
    ax.text(
        0.98, 0.06,
        rf"$\begin{{aligned}}"
        # 1) Omega_m
        r"\Omega_{{\rm m}}\,&=" + f"{align_str['Omega_m']:.2f}" + r"\\" 
        # 2) m_WDM
        r"m_{{\rm WDM}}\,&=" + f"{align_str['m_wdm']:.2f}" + r"\\" 
        # 3) T_vir
        r"T_{{\rm vir}}\,&=10^{" + f"{align_str['Tvir_exp']:.2f}" + r"}\\" 
        # 4) zeta
        r"\zeta\,&=" + f"{align_str['zeta']:.2f}" + r"\\" 
        # 5) E0
        r"E_0\,&=" + f"{align_str['E0']:.2f}" + r"\\" 
        # 6) L_X
        r"L_X\,&=10^{" + f"{align_str['LX_exp']:.2f}" + r"}"
        r"\end{aligned}$",
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=30,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    """

    ax.set_xlabel(r'$z$', fontsize=45)
    ax.set_ylabel(r'$x_{\mathrm{HI}}$', fontsize=45)
    ax.set_xlim(zs.min(), zs.max())
    ax.set_ylim(-0.05, 1.0)
    ax.tick_params(labelsize=36)
    ax.legend(fontsize=35, loc='lower right')
    fig.subplots_adjust(left=0.10, right=0.88, top=0.95, bottom=0.12)

    fig.savefig(out_file, dpi=400)
    plt.close(fig)


def main():
    pure_mod  = load_model(pure_model_dir)
    noise_mod = load_model(noise_model_dir)

    ds_pure = PowerSpectrumDataset(
        pure_data_dir,
        max_ones_allowed=dataset_params['max_ones_allowed'],
        max_zeros_allowed=dataset_params['max_zeros_allowed'],
        filter_reionization_timing=dataset_params['filter_reionization_timing'],
        min_redshift_index=dataset_params['min_redshift_index'],
        max_redshift_index=dataset_params['max_redshift_index'],
        add_noise=dataset_params['add_noise']
    )
    all_files = [os.path.basename(p) for p in ds_pure.files]

    if specific_file:
        chosen = [specific_file]
    else:
        chosen = list(np.random.choice(all_files,
                                       size=min(num_random, len(all_files)),
                                       replace=False))

    for fname in chosen:
        logging.info(f"Processing {fname}")
        # load parameters
        fullpath      = next(p for p in ds_pure.files if os.path.basename(p)==fname)
        m_wdm, Omega_m, E0, LX_exp, Tvir_exp, zeta = np.load(fullpath)['params']
        align_vals = {
            'Omega_m': Omega_m,
            'm_wdm':  m_wdm,
            'E0':      E0,
            'LX_exp':  LX_exp,
            'Tvir_exp':Tvir_exp,
            'zeta':    zeta
        }

        # find matching noise file
        base, ext = os.path.splitext(fname)
        ds_noise   = PowerSpectrumDataset(noise_data_dir, max_ones_allowed=dataset_params['max_ones_allowed'],
        max_zeros_allowed=dataset_params['max_zeros_allowed'],
        filter_reionization_timing=dataset_params['filter_reionization_timing'],
        min_redshift_index=dataset_params['min_redshift_index'],
        max_redshift_index=dataset_params['max_redshift_index'],
        add_noise=dataset_params['add_noise'])
        match      = next(
            (os.path.basename(p) for p in ds_noise.files
             if os.path.splitext(os.path.basename(p))[0] in {base, base + "_noisy"}),
            None
        )
        if not match:
            logging.warning(f"No noise match for {fname}, skipping")
            continue

        zs, th, spure = evaluate_model_on_file(pure_mod, pure_data_dir, fname)
        _, _, snoise  = evaluate_model_on_file(noise_mod, noise_data_dir, match)
        mp, sp        = spure.mean(axis=0), spure.std(axis=0)
        mn, sn        = snoise.mean(axis=0), snoise.std(axis=0)

        out = os.path.join(output_dir, f"history_{base}.pdf")
        plot_history(zs, mp, sp, mn, sn, th, align_vals, out, plot_data=plot_data)
        logging.info(f"Saved {out}")


if __name__ == '__main__':
    main()