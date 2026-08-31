"""Mutual information estimator for EoRFlow-dev inference models.

This module estimates the mutual information I(theta; x) by Monte-Carlo
using samples from the dataset (which represent draws from the prior and
forward model). The estimator uses

    Y_i = log q(theta_i | x_i) - log p(theta_i)

and returns MI = mean(Y_i) with a standard error computed from the sample
variance.

Supports both the FrEIA flow wrapper (`ConditionalInvertibleBlock`) and the
conditional flow-matching model (`ConditionalCFM`) by calling their
`log_prob(...)` methods. The empirical prior p(theta) can be estimated from
the dataset using a multivariate Gaussian fit, independent per-dimension KDE
or a simple histogram fallback.

This file is designed to integrate with the project's existing dataset and
model classes (see `sample.py` for conventions).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

# project imports (relative layout used elsewhere in project)
sys.path.append('/pfs/10/work/hd_pt254-eorflow/EoRFlow-dev/src/models')
sys.path.append('/pfs/10/work/hd_pt254-eorflow/EoRFlow-dev/src/data_tools')
from flow import ConditionalInvertibleBlock
from cfm import ConditionalCFM
from data_loader import EoRH5Dataset, SkatrGridDataset


def estimate_empirical_prior(labels: np.ndarray, method: str = 'kde') -> Callable[[np.ndarray], np.ndarray]:
    """Estimate an empirical prior log-probability function from labels.

    labels: array shape (N, D)
    method: one of {'kde', 'gauss', 'hist'}

    Returns a function prior_log_prob(theta_array) -> log_probs (shape (M,))
    which expects theta_array in the same parameter space as `labels`.
    """
    assert labels.ndim == 2
    N, D = labels.shape

    if method == 'gauss':
        mu = labels.mean(axis=0)
        cov = np.cov(labels, rowvar=False)
        # regularize covariance
        cov += np.eye(D) * 1e-6
        cov_inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)

        norm_const = -0.5 * (D * math.log(2 * math.pi) + logdet)

        def prior_log_prob(theta: np.ndarray) -> np.ndarray:
            # theta: (M, D) or (D,)
            t = np.atleast_2d(theta)
            diff = t - mu
            exp = -0.5 * np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
            return exp + norm_const

        return prior_log_prob

    if method == 'kde':
        try:
            from scipy.stats import gaussian_kde
            kde_available = True
        except Exception:
            kde_available = False

        if kde_available:
            # build one KDE per-dimension and approximate joint as product of marginals
            kdes = []
            for d in range(D):
                arr = labels[:, d]
                # handle degenerate
                if np.allclose(arr, arr[0]):
                    kdes.append(None)
                else:
                    kdes.append(gaussian_kde(arr))

            def prior_log_prob(theta: np.ndarray) -> np.ndarray:
                t = np.atleast_2d(theta)
                M = t.shape[0]
                logs = np.zeros(M, dtype=float)
                for d in range(D):
                    arr = t[:, d]
                    if kdes[d] is None:
                        # delta at single value
                        val = labels[0, d]
                        logs += np.log((arr == val).astype(float) + 1e-300)
                    else:
                        pdf = np.maximum(kdes[d].evaluate(arr), 1e-300)
                        logs += np.log(pdf)
                return logs

            return prior_log_prob

        # fallback to per-dimension histograms
        method = 'hist'

    if method == 'hist':
        bins_list = []
        hist_list = []
        bin_edges = []
        for d in range(D):
            arr = labels[:, d]
            edges = np.histogram_bin_edges(arr, bins='auto')
            hist, _ = np.histogram(arr, bins=edges, density=True)
            # avoid zero bins by floor
            hist = np.maximum(hist, 1e-12)
            bins_list.append(edges)
            hist_list.append(hist)
            bin_edges.append(edges)

        def prior_log_prob(theta: np.ndarray) -> np.ndarray:
            t = np.atleast_2d(theta)
            M = t.shape[0]
            logs = np.zeros(M, dtype=float)
            for d in range(D):
                edges = bins_list[d]
                hist = hist_list[d]
                inds = np.searchsorted(edges, t[:, d], side='right') - 1
                inds = np.clip(inds, 0, len(hist) - 1)
                pdf = hist[inds]
                logs += np.log(np.maximum(pdf, 1e-300))
            return logs

        return prior_log_prob

    raise ValueError(f'Unknown prior estimation method: {method}')


def compute_mutual_information(model, dataset, n_total: int = 1000, prior_method: str = 'kde', device: torch.device | None = None, batch_size: int = 32, seed: int = 42) -> dict:
    """Compute Monte-Carlo estimate of mutual information I(theta; x).

    - dataset: a torch.utils.data.Dataset that returns (cond, label) pairs
      where `label` is the parameter vector theta in the same space the
      model expects for `log_prob`.
    - model: object implementing `log_prob(theta_tensor, cond_tensor)` or
      `log_prob(theta_tensor, data_tensor)` (both supported). The function
      will call it with torch.no_grad() on `device`.
    """
    rng = np.random.default_rng(seed)
    total_ds = len(dataset)
    if n_total > total_ds:
        raise ValueError(f'n_total ({n_total}) larger than dataset size ({total_ds})')

    idxs = rng.choice(total_ds, size=n_total, replace=False)

    # Gather labels to fit prior
    labels = []
    for i in idxs:
        c, y = dataset[i]
        labels.append(y.numpy())
    labels = np.stack(labels, axis=0)  # (n_total, D)

    prior_logfn = estimate_empirical_prior(labels, method=prior_method)

    # batch evaluation
    device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    y_sum = 0.0
    y2_sum = 0.0

    D = labels.shape[1]
    # iterate in batches
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        bidx = idxs[start:end]
        conds = []
        thetas = []
        for i in bidx:
            c, y = dataset[i]
            conds.append(c.numpy())
            thetas.append(y.numpy())
        conds = np.stack(conds, axis=0)
        thetas = np.stack(thetas, axis=0)

        # convert to torch and move to device
        with torch.no_grad():
            t_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)
            c_tensor = torch.tensor(conds, dtype=torch.float32, device=device)

            # call model.log_prob. The models in project accept signatures
            # either (theta, data) or (x, c). We'll try both.
            try:
                q_log = model.log_prob(t_tensor, c_tensor)
            except TypeError:
                # try swapping
                q_log = model.log_prob(t_tensor, c_tensor)

            # ensure numpy
            q_log_np = q_log.cpu().numpy().ravel()

        prior_log_np = prior_logfn(thetas)

        y_batch = q_log_np - prior_log_np
        y_sum += y_batch.sum()
        y2_sum += (y_batch**2).sum()

    n_total = float(n_total)
    mi = float(y_sum / n_total)
    # sample variance
    sample_var = float((y2_sum - n_total * mi * mi) / (n_total - 1.0)) if n_total > 1 else 0.0
    mi_se = math.sqrt(max(sample_var, 0.0) / n_total)

    return dict(mi_nats=mi, mi_bits=mi / math.log(2), mi_se_nats=mi_se, mi_se_bits=mi_se / math.log(2), sample_var=sample_var)


def find_latest_model_dir(project_root, output_subpath, mode, data):
    base = Path(project_root) / output_subpath / mode / data
    if not base.exists():
        raise FileNotFoundError(f"Output base not found: {base}")
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if (d / 'best_flow_model.pth').exists() or (d / 'final_flow_model.pth').exists() or (d / 'best_cnn_model.pth').exists():
            candidates.append(d)
    if not candidates:
        for d in base.rglob('*'):
            if d.is_dir() and ((d / 'best_flow_model.pth').exists() or (d / 'final_flow_model.pth').exists() or (d / 'best_cnn_model.pth').exists()):
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No trained model directories found under {base}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', choices=['ps2d','ps1d','cnn','skatr'], default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, help='Path to model folder or checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of dataset draws to use for MC')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--prior_method', type=str, choices=['kde','gauss','hist'], default='kde')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # minimal config merging (keep behaviour similar to sample.py)
    cfg = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as fh:
            cfg = yaml.safe_load(fh) or {}

    project_root = cfg.get('project_root', '/pfs/10/work/hd_pt254-eorflow')
    output_subpath = cfg.get('output_subpath', 'EoRFlow-dev/output')
    model_arg = args.model or cfg.get('model_path', None)
    mode = args.mode or cfg.get('mode', 'ps2d')
    data = args.data or cfg.get('data', 'opt_noise')
    data_type = cfg.get('data_type', ('skatr' if mode == 'skatr' else 'standard'))
    learn_target = cfg.get('learn_target', 'xhi')
    print(f'Using mode: {mode}, data: {data}')
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # find model dir
    if model_arg:
        model_path = Path(model_arg)
        if model_path.is_file():
            model_dir = model_path.parent
        else:
            model_dir = model_path
    else:
        model_dir = find_latest_model_dir(project_root, output_subpath, mode, data)

    best_cfm = model_dir / 'best_cfm_model.pth'
    final_cfm = model_dir / 'final_cfm_model.pth'
    best_flow = model_dir / 'best_flow_model.pth'
    final_flow = model_dir / 'final_flow_model.pth'
    best_joint = model_dir / 'best_joint.pth'

    flow_ckpt = None
    detected_backend = None
    if best_cfm.exists():
        flow_ckpt = best_cfm; detected_backend = 'cfm'
    elif final_cfm.exists():
        flow_ckpt = final_cfm; detected_backend = 'cfm'
    elif best_flow.exists():
        flow_ckpt = best_flow; detected_backend = 'flow'
    elif final_flow.exists():
        flow_ckpt = final_flow; detected_backend = 'flow'
    elif best_joint.exists():
        ck = torch.load(str(best_joint), map_location='cpu')
        if isinstance(ck, dict) and 'inference_state_dict' in ck:
            flow_ckpt = best_joint; detected_backend = 'cfm'
        else:
            flow_ckpt = best_joint; detected_backend = 'flow'
    else:
        flow_ckpt = None; detected_backend = None

    # build model instance
    min_redshift_index = int(cfg.get('min_redshift_index', 0))
    max_redshift_index = int(cfg.get('max_redshift_index', 15))
    redshift_dim = max_redshift_index - min_redshift_index
    target_dim = redshift_dim
    skcfg = cfg.get('skatr', {})

    # infer dimensions for SKATR representation
    cond_dims = None
    if data_type == 'skatr' or mode == 'skatr':
        source = skcfg.get('source', data)
        source_dirs = skcfg.get('source_dirs', {})
        skatr_data_dirs = skcfg.get('data_dirs', source_dirs.get(source, []))
        if isinstance(skatr_data_dirs, str):
            skatr_data_dirs = [skatr_data_dirs]
        if not skatr_data_dirs:
            raise ValueError("No SKATR data directories configured. Set skatr.data_dirs or skatr.source_dirs[skatr.source].")

        ds_probe = SkatrGridDataset(
            data_dirs=skatr_data_dirs,
            target=learn_target,
            logit=(learn_target == 'xhi'),
            min_redshift_index=min_redshift_index,
            max_redshift_index=max_redshift_index,
            sim_param_indices=skcfg.get('sim_param_indices', None),
            drop_tvir=skcfg.get('drop_tvir', True),
            num_sim_params=skcfg.get('num_sim_params', 5),
            normalize_cond=skcfg.get('normalize_cond', False),
        )
        c0, y0 = ds_probe[0]
        cond_dims = int(c0.numel())
        target_dim = int(y0.numel())

    # compute cond_dims as sample.py defaults
    if cond_dims is not None:
        pass
    elif mode == 'cnn':
        cond_dims = redshift_dim
    elif mode == 'ps2d':
        obs_dim = redshift_dim * 10 * 10
        cond_dims = obs_dim + redshift_dim
    elif mode == 'ps1d':
        obs_dim = redshift_dim * 14
        cond_dims = obs_dim + redshift_dim
    elif mode == 'skatr':
        cond_dims = 360
    else:
        raise ValueError(f'Unknown mode: {mode}')

    model = None
    if detected_backend == 'cfm':
        cfm_n_layers = cfg.get('cfm', {}).get('n_layers', 3)
        cfm_hidden = cfg.get('cfm', {}).get('hidden_dim', 512)
        cfm_alpha = cfg.get('cfm', {}).get('alpha', 0.0)
        cfm_pdrop = cfg.get('cfm', {}).get('p_drop', 0.0)
        model = ConditionalCFM(n_dim=target_dim, summary_dim=cond_dims, n_layers=cfm_n_layers, hidden_dim=cfm_hidden, alpha=cfm_alpha, p_drop=cfm_pdrop)
        model.to(device)
        if flow_ckpt:
            ck = torch.load(str(flow_ckpt), map_location=device)
            if isinstance(ck, dict) and 'inference_state_dict' in ck:
                model.load_state_dict(ck['inference_state_dict'])
            elif isinstance(ck, dict) and 'flow' in ck and isinstance(ck['flow'], dict):
                model.load_state_dict(ck['flow'])
            else:
                model.load_state_dict(ck)
    else:
        # default to FrEIA wrapper
        params = {'flow': {'n_dim': target_dim, 'n_blocks': cfg.get('n_blocks', 10), 'n_nodes': cfg.get('n_nodes', 512), 'cond_dims': cond_dims, 'subnet_depth': cfg.get('subnet_depth', 2), 'act': cfg.get('act', 'relu'), 'load': False, 'model_location': None}}
        model = ConditionalInvertibleBlock(params)
        model.to(device)
        if flow_ckpt:
            # try wrapper's load_model first (some wrappers implement it)
            loaded = False
            try:
                loaded = model.load_model(str(flow_ckpt))
            except Exception:
                loaded = False

            if not loaded:
                try:
                    sd = torch.load(str(flow_ckpt), map_location=device)
                    # pick nested dict if present
                    if isinstance(sd, dict) and 'flow' in sd and isinstance(sd['flow'], dict):
                        sd_part = sd['flow']
                    elif isinstance(sd, dict) and 'inference_state_dict' in sd and isinstance(sd['inference_state_dict'], dict):
                        sd_part = sd['inference_state_dict']
                    else:
                        sd_part = sd
                    # attempt non-strict load to allow partially compatible checkpoints
                    try:
                        model.load_state_dict(sd_part, strict=False)
                        logging.info(f'Loaded flow checkpoint (partial non-strict) from {flow_ckpt}')
                    except Exception as e:
                        # final fallback: try strict load to raise informative error
                        model.load_state_dict(sd_part)
                except Exception as e:
                    raise FileNotFoundError(f'Failed to load flow checkpoint: {flow_ckpt} -> {e}')

    # prepare dataset
    db_subpath = cfg.get('database_subpath', 'database')
    data_dirs = [os.path.join(project_root, db_subpath, data, f'batch_0')]
    if data_type == 'skatr' or mode == 'skatr':
        print(f'Using SKATR data dirs: {skatr_data_dirs}')
        ds = SkatrGridDataset(
            data_dirs=skatr_data_dirs,
            target=learn_target,
            logit=(learn_target == 'xhi'),
            min_redshift_index=min_redshift_index,
            max_redshift_index=max_redshift_index,
            sim_param_indices=skcfg.get('sim_param_indices', None),
            drop_tvir=skcfg.get('drop_tvir', True),
            num_sim_params=skcfg.get('num_sim_params', 5),
            normalize_cond=skcfg.get('normalize_cond', False),
        )
    else:
        print(f'Using data dirs: {data_dirs}')
        ds = EoRH5Dataset(
            data_dirs=data_dirs,
            mode=mode,
            min_redshift_index=min_redshift_index,
            max_redshift_index=max_redshift_index,
            logit=True,
            num_files=cfg.get('num_files', None),
        )

    # logger
    logfile = os.path.join(model_dir, 'mutual_information.log')
    logger = logging.getLogger('mutual_information')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    logger.info(
        f'Using model dir: {model_dir}, backend: {detected_backend}, dataset size: {len(ds)}, '
        f'data_type={data_type}, learn_target={learn_target}, target_dim={target_dim}, cond_dims={cond_dims}'
    )

    res = compute_mutual_information(model, ds, n_total=args.num_samples, prior_method=args.prior_method, device=device, batch_size=args.batch_size)

    logger.info(f"Estimated MI: {res['mi_nats']:.6f} nats ({res['mi_bits']:.6f} bits)")
    logger.info(f"Std error: {res['mi_se_nats']:.6e} nats ({res['mi_se_bits']:.6e} bits)")
    logger.info(f"Sample variance (nats^2): {res['sample_var']:.6e}")
    print(f"Estimated MI: {res['mi_nats']:.6f} nats ({res['mi_bits']:.6f} bits)")
    print(f"Std error: {res['mi_se_nats']:.6e} nats ({res['mi_se_bits']:.6e} bits)")
    print(f"Log written to: {logfile}")


if __name__ == '__main__':
    main()
