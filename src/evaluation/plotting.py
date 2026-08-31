"""Plotting utilities for EoRFlow.

Functions:
 - load_samples: load samples.npz
 - plot_scatter_grid: scatter mean vs truth with errorbars per redshift
 - plot_tarp_coverage: compute and plot TARP coverage (uses get_tarp_coverage from tarp)
 - plot_corner_and_hist: create corner plots for selected redshifts
 - plot_eor_history: plot predicted mean and CI vs redshift for chosen sims

CLI:
 The module provides a main() that reads a samples file, a few options, and
 writes PDFs into a `plots/` directory next to the samples file.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
from typing import Sequence

# notebook imports
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

try:
    from tarp import get_tarp_coverage
except Exception:
    get_tarp_coverage = None

try:
    import corner
except Exception:
    corner = None


EORFLOW_REDSHIFTS = np.array([
    5.233072, 5.4996195, 6.0014706, 6.4982133, 7.00021, 7.5001707,
    8.000271, 8.499395, 8.997384, 9.501907, 10.002276, 10.500712,
    11.000936, 11.499588, 12.002204,
], dtype=np.float32)


def load_samples(samples_path: str):
    data = np.load(samples_path, allow_pickle=True)
    preds = data['preds']
    labels = data['labels'] if 'labels' in data else None
    conds = data['conds'] if 'conds' in data else None
    info = {}
    # Unpack top-level scalar/object fields.
    for k in data.files:
        if k in ('preds', 'labels', 'conds'):
            continue
        v = data[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            v = v.item()
        info[k] = v

    # If sampler stored metadata under key "info", flatten it for easier access.
    if 'info' in info and isinstance(info['info'], dict):
        nested = info.pop('info')
        info.update(nested)

    return preds, labels, conds, info


def ensure_plots_dir(samples_path: str) -> str:
    out_dir = os.path.join(os.path.dirname(samples_path), 'plots')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_scatter_grid(preds: np.ndarray, labels: np.ndarray, outpath: str, dim_labels: Sequence[str] | None = None, learn_target: str | None = None):
    # preds: (S, N, D) ; labels: (N, D)
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)

    n_red = pred_mean.shape[1]
    if dim_labels is None:
        dim_labels = [f'dim_{i}' for i in range(n_red)]
    cols = 3
    rows = int(np.ceil(n_red / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), squeeze=False)

    for i in range(n_red):
        r = i // cols
        c = i % cols
        ax = axs[r][c]

        x = labels[:, i]
        y = pred_mean[:, i]
        yerr = pred_std[:, i]

        jitter = (np.random.RandomState(0).rand(len(x)) - 0.5) * 0.0
        ax.errorbar(x + jitter, y, yerr=yerr, fmt='o', ms=4, alpha=0.6, color='C0', elinewidth=0.8, capsize=2)

        mn = np.nanmin(np.concatenate([x, y]))
        mx = np.nanmax(np.concatenate([x, y]))
        pad = 0.02 * (mx - mn if mx > mn else 1.0)
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls='--', color='k', lw=0.8)

        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        span = hi - lo
        pad_lim = 0.05 * span if span > 0 else 1.0
        ax.set_xlim(lo - pad_lim, hi + pad_lim)
        ax.set_ylim(lo - pad_lim, hi + pad_lim)
        if learn_target == 'xhi':
            ax.set_xlabel(r'True $x_{\mathrm{HI}}$', fontsize=20)
            ax.set_ylabel(r'Pred mean $x_{\mathrm{HI}}$', fontsize=20)
        else:
            ax.set_xlabel('True', fontsize=20)
            ax.set_ylabel('Pred mean', fontsize=20)
        title = dim_labels[i] if i < len(dim_labels) else f'dim_{i}'
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    total = rows * cols
    for k in range(n_red, total):
        r = k // cols
        c = k % cols
        axs[r][c].axis('off')

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def plot_tarp_coverage(preds: np.ndarray, labels: np.ndarray, outpath: str):
    if get_tarp_coverage is None:
        raise RuntimeError('tarp.get_tarp_coverage not available in environment')

    ecp_bootstrap, alpha_bootstrap = get_tarp_coverage(preds, labels, references="random", metric="euclidean", norm=True, bootstrap=True)
    k_sigma = [1, 2, 3]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls='--', color='k', label="Ideal case")
    ax.plot(alpha_bootstrap, ecp_bootstrap.mean(axis=0), label='TARP')
    for k in k_sigma:
        ax.fill_between(alpha_bootstrap, ecp_bootstrap.mean(axis=0) - k * ecp_bootstrap.std(axis=0), ecp_bootstrap.mean(axis=0) + k * ecp_bootstrap.std(axis=0), alpha=0.2)
    ax.legend()
    ax.set_ylabel("Expected Coverage")
    ax.set_xlabel("Credibility Level")
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def plot_corner_and_hist(preds: np.ndarray, labels: np.ndarray, outpath: str, n_panels: int = 5, redshift_idxs: Sequence[int] | None = None, example_sims: Sequence[int] | None = None, dim_labels: Sequence[str] | None = None, prior_widths: Sequence[float] | None = None, posterior_volume: dict | None = None):
    # preds: (S, N, D)
    n_red = preds.shape[2] if preds.ndim == 3 else preds.shape[1]
    if redshift_idxs is None:
        use_panels = min(n_panels, n_red)
        redshift_idxs = np.linspace(0, n_red - 1, use_panels, dtype=int)
    if example_sims is None:
        example_sims = np.arange(min(1, preds.shape[1]))

    if dim_labels is None:
        dim_labels = [f'dim_{ri}' for ri in redshift_idxs]
    elif len(dim_labels) == n_red:
        # If full-dimension labels were passed, select the matching subset for corner.
        ridx = np.asarray(redshift_idxs, dtype=int)
        dim_labels = [dim_labels[i] for i in ridx]

    if prior_widths is not None:
        prior_widths = np.asarray(prior_widths, dtype=float)
        if prior_widths.shape[0] != n_red:
            prior_widths = None

    labels_sel = labels[:, redshift_idxs]
    preds_sel = preds[:, :, redshift_idxs]

    # Build summary text once. Prefer sampler-provided posterior-volume stats.
    summary_lines = None
    if isinstance(posterior_volume, dict):
        vals = posterior_volume.get('per_param_relative_uncertainty')
        if vals is not None:
            arr = np.asarray(vals, dtype=float)
            idx = np.asarray(redshift_idxs, dtype=int)
            if arr.shape[0] >= np.max(idx) + 1:
                shown = arr[idx]
                summary_lines = [r'$\langle\sigma_{\mathrm{post}}\rangle/\sigma_{\mathrm{test}}$']
                for j, name in enumerate(dim_labels):
                    v = shown[j]
                    summary_lines.append(f'{name}: {v:.3f}' if np.isfinite(v) else f'{name}: n/a')
        

    clr = 'purple'
    for sim in example_sims:
        samples = preds_sel[:, sim, :]
        truth = labels_sel[sim, :]

        if corner is None:
            raise RuntimeError('corner package not available')

        fig = corner.corner(
            samples,
            bins=10,
            labels=dim_labels,
            truths=truth,
            truth_color='purple',
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={'fontsize': 20},
            label_kwargs={'fontsize': 20},
            labelpad=0.1,
            quantiles=[0.16, 0.5, 0.84],
            levels=(0.68, 0.95),
            smooth=1.2,
            smooth1d=1.2,
            plot_datapoints=True,
            plot_density=True,
            fill_contours=True,
            data_kwargs={'alpha':0.35, 'color':clr},
            contour_kwargs={'colors':[clr], 'linewidths':1.0},
            hist_kwargs={'color':clr, 'alpha':0.6},
            reverse=False,
            use_math_text=True,
            max_n_ticks=3,
        )
        for ax in fig.get_axes():
            ax.tick_params(axis='both', which='major', labelsize=20)

        if summary_lines is None:
            # Backward-compatible fallback for legacy samples without metadata.
            # Equivalent to compute_relative_uncertainty on selected dimensions.
            sample_std = np.std(preds_sel, axis=0)
            mean_posterior_std = np.mean(sample_std, axis=0)
            label_std = np.std(labels_sel, axis=0)
            frac = mean_posterior_std / (label_std + 1e-10)

            lines = [r'$\langle\sigma_{\mathrm{post}}\rangle/\sigma_{\mathrm{test}}$']
            for j, name in enumerate(dim_labels):
                val = frac[j]
                if np.isfinite(val):
                    lines.append(f'{name}: {val:.3f}')
                else:
                    lines.append(f'{name}: n/a')

        else:
            lines = summary_lines

        fig.text(
            0.985,
            0.985,
            '\n'.join(lines),
            ha='right',
            va='top',
            fontsize=20,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='0.5'),
        )

        fig.savefig(outpath.replace('.pdf', f'_sim{sim}.pdf'), bbox_inches='tight')
        plt.close(fig)


def plot_eor_history(preds: np.ndarray, labels: np.ndarray, outpath: str, example_sims: Sequence[int] | None = None):
    if example_sims is None:
        example_sims = [0]

    n_dim = preds.shape[2] if preds.ndim == 3 else preds.shape[1]
    zs = np.linspace(5., 12., n_dim)

    for sim in example_sims:
        mean_pred = np.mean(preds[:, sim, :], axis=0)
        std_pred = np.std(preds[:, sim, :], axis=0)
        truth = labels[sim, :]

        plt.figure(figsize=(9, 4))
        plt.plot(zs, mean_pred, color='purple', lw=2, label='Predicted mean')
        plt.fill_between(zs, mean_pred - std_pred, mean_pred + std_pred, color='purple', alpha=0.25, label='68% CI')
        plt.plot(zs, truth, color='k', lw=1.5, ls='--', label='Truth')
        plt.xlabel('z', fontsize=20)
        plt.ylabel('xHI', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='lower right')
        plt.savefig(outpath.replace('.pdf', f'_sim{sim}.pdf'), bbox_inches='tight')
        plt.close()


def _default_param_names_mathtex(data_name: str | None, n_dim: int):
    """Return default sim-parameter names in mathtext for known datasets."""
    data_name = (data_name or '').lower()

    # 21cmFAST canonical order includes log10 T_vir as first parameter.
    # For current SKATR use, we usually skip the first one and keep 6 params.
    names_21cmfast_full = [
        r'$\log_{10}T_{\mathrm{vir}}$',
        r'$f_{\mathrm{esc},10}$',
        r'$f_{\star,10}$',
        r'$\alpha_{\mathrm{esc}}$',
        r'$\alpha_{\star}$',
        r'$\log_{10}L_X$',
        r'$\zeta$',
    ]

    # If dimension indicates first param was dropped, use the trailing 6.
    if data_name in ('pure', '21cmfast', '21cm_fast'):
        if n_dim == 6:
            return names_21cmfast_full[1:]
        if n_dim <= len(names_21cmfast_full):
            return names_21cmfast_full[:n_dim]

    # Loreli parameter names
    names_loreli = [
        r'$f_x$',
        r'$\tau$',
        r'$r_H$',
        r'$\log M_{\min}$',
        r'$f_{\mathrm{esc,post}}$',
    ]
    if data_name == 'loreli':
        return names_loreli[:n_dim]

    return None


def _default_xhi_labels_mathtex(n_dim: int, data_name: str | None = None):
    data_name = (data_name or '').lower()

    # Loreli setup commonly uses the z=6.5..12 subset (12 bins).
    if data_name == 'loreli' and n_dim == 12:
        zs = EORFLOW_REDSHIFTS[3:15]
    elif n_dim <= len(EORFLOW_REDSHIFTS):
        zs = EORFLOW_REDSHIFTS[:n_dim]
    else:
        zs = np.linspace(5.0, 12.0, n_dim, dtype=np.float32)

    return [rf'$x_{{\mathrm{{HI}}}}(z={z:.2f})$' for z in zs]


def _default_prior_widths(data_name: str | None, n_dim: int):
    data_name = (data_name or '').lower()
    if data_name == 'loreli':
        vals = np.array([9.9, 9765.8, 1.0, 1.742176558101871, 0.45], dtype=float)
        return vals[:n_dim] if n_dim <= vals.shape[0] else None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'eval.yaml'), help='Path to eval.yaml with plotting defaults')
    parser.add_argument('--samples', '-s', help='Path to samples.npz (overrides config)')
    parser.add_argument('--mode', help='mode name for organizing output')
    parser.add_argument('--data', help='data name for organizing output')
    parser.add_argument('--n_panels', type=int, help='Number of panels for corner')
    parser.add_argument('--redshift_idxs', nargs='*', type=int, help='Which redshift indices to show in corner')
    parser.add_argument('--example_sims', nargs='*', type=int, help='Which example simulations to show (defaults to 1)')
    args = parser.parse_args()

    # load config and merge with CLI args (CLI overrides)
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    plotting_cfg = cfg.get('plotting', {}) if isinstance(cfg, dict) else {}

    samples_path = args.samples or plotting_cfg.get('samples_file')
    preds, labels, conds, info = load_samples(samples_path)
    outdir = ensure_plots_dir(samples_path)
    info = info or {}
    
    # Extract dataset name from samples filename (samples_{data}.npz format)
    samples_basename = os.path.basename(samples_path)
    if samples_basename.startswith('samples_') and samples_basename.endswith('.npz'):
        data_name = samples_basename.replace('samples_', '').replace('.npz', '')
    else:
        data_name = 'unknown'

    logger = logging.getLogger('plotting')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Determine merged settings
    mode = args.mode or plotting_cfg.get('mode', 'ps2d')
    data = args.data or plotting_cfg.get('data', 'opt_noise')
    n_panels = args.n_panels if args.n_panels is not None else plotting_cfg.get('n_panels', 5)
    red_idxs = args.redshift_idxs if args.redshift_idxs is not None else plotting_cfg.get('redshift_idxs')
    sims = args.example_sims if args.example_sims is not None else plotting_cfg.get('example_sims')
    learn_target = info.get('learn_target', 'xhi')

    n_dim = labels.shape[1]
    # Parameter names can come from:
    # 1) samples info (preferred if stored at sampling time),
    # 2) plotting.param_names in config,
    # 3) top-level param_names in config,
    # 4) skatr.param_names in config,
    # 5) built-in defaults for known datasets,
    # then fall back to generic dim labels.
    param_names = info.get('param_names')

    # For xHI targets, use redshift-aware mathtext labels by default.
    if learn_target == 'xhi' and param_names is None:
        ds_name = args.data or plotting_cfg.get('data') or (cfg.get('data') if isinstance(cfg, dict) else None)
        param_names = _default_xhi_labels_mathtex(n_dim, ds_name)
    if param_names is None:
        param_names = plotting_cfg.get('param_names')
    if param_names is None:
        param_names = cfg.get('param_names') if isinstance(cfg, dict) else None
    if param_names is None and isinstance(cfg, dict):
        param_names = (cfg.get('skatr', {}) or {}).get('param_names')
    if param_names is None:
        # Prefer explicit CLI/config data name; fallback to metadata.
        ds_name = args.data or plotting_cfg.get('data') or cfg.get('data') if isinstance(cfg, dict) else None
        param_names = _default_param_names_mathtex(ds_name, n_dim)

    if param_names is None:
        base_labels = [f'dim_{i}' for i in range(n_dim)]
    else:
        base_labels = [str(x) for x in list(param_names)]
        if len(base_labels) < n_dim:
            base_labels = base_labels + [f'dim_{i}' for i in range(len(base_labels), n_dim)]
        elif len(base_labels) > n_dim:
            base_labels = base_labels[:n_dim]

    if red_idxs is not None:
        dim_labels = [base_labels[ri] if 0 <= ri < len(base_labels) else f'dim_{ri}' for ri in red_idxs]
    else:
        dim_labels = base_labels

    # scatter grid
    scatter_pdf = os.path.join(outdir, f'scatter_{data_name}.pdf')
    plot_scatter_grid(preds, labels, scatter_pdf, dim_labels=base_labels, learn_target=learn_target)
    logger.info(f'Wrote {scatter_pdf}')

    # tarp coverage (safe to skip if tarp not installed)
    try:
        tarp_pdf = os.path.join(outdir, f'tarp_coverage_{data_name}.pdf')
        plot_tarp_coverage(preds, labels, tarp_pdf)
        logger.info(f'Wrote {tarp_pdf}')
    except Exception as e:
        logger.info(f'Skipping TARP plot: {e}')

    # Resolve prior widths for fallback posterior-volume annotation in corner plots.
    prior_widths = plotting_cfg.get('prior_widths')
    if prior_widths is None and isinstance(cfg, dict):
        prior_widths = cfg.get('prior_widths')
    if prior_widths is None and isinstance(cfg, dict):
        skatr_cfg = cfg.get('skatr', {}) or {}
        prior_widths = skatr_cfg.get('prior_widths')
        if prior_widths is None and skatr_cfg.get('prior_min') is not None and skatr_cfg.get('prior_max') is not None:
            prior_min = np.asarray(skatr_cfg.get('prior_min'), dtype=float)
            prior_max = np.asarray(skatr_cfg.get('prior_max'), dtype=float)
            if prior_min.shape == prior_max.shape:
                prior_widths = (prior_max - prior_min).tolist()
    if prior_widths is None:
        ds_name = args.data or plotting_cfg.get('data') or (cfg.get('data') if isinstance(cfg, dict) else None)
        defaults = _default_prior_widths(ds_name, n_dim)
        if defaults is not None:
            prior_widths = defaults.tolist()
    if prior_widths is not None:
        prior_widths = [float(x) for x in list(prior_widths)[:n_dim]]
        if len(prior_widths) < n_dim:
            prior_widths = prior_widths + [np.nan] * (n_dim - len(prior_widths))

    posterior_volume = info.get('posterior_volume') if isinstance(info, dict) else None

    # corner and eor history
    corner_pdf = os.path.join(outdir, f'corner_{data_name}.pdf')
    plot_corner_and_hist(
        preds,
        labels,
        corner_pdf,
        n_panels=n_panels,
        redshift_idxs=red_idxs,
        example_sims=sims,
        dim_labels=dim_labels,
        prior_widths=prior_widths,
        posterior_volume=posterior_volume,
    )
    logger.info(f'Wrote corner PDFs in {outdir}')

    if learn_target == 'xhi':
        eor_pdf = os.path.join(outdir, f'eor_history_{data_name}.pdf')
        plot_eor_history(preds, labels, eor_pdf, example_sims=sims)
        logger.info(f'Wrote EoR history PDFs in {outdir}')
    else:
        logger.info(f"Skipping EoR history plot for learn_target={learn_target}")


if __name__ == '__main__':
    main()
