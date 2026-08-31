#!/usr/bin/env python
"""Sampler for EoRFlow-dev models.

Generates posterior samples for modes: ps2d, ps1d, cnn, skatr and saves as npz.
Default output shape: (samples_per_sim, num_sims, param_dim)

Reads options from YAML (see config/eval.yaml) or CLI. If model path is not
provided the script will try to find the latest trained model under the
output_subpath for the chosen mode/data.
"""
import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

# project imports
sys.path.append('/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/src/models')
sys.path.append('/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/src/data_tools')
from flow import ConditionalInvertibleBlock
from cfm import ConditionalCFM
from data_loader import EoRH5Dataset, SkatrGridDataset
from cnn import ConvNet3D


def find_latest_model_dir(project_root, output_subpath, mode, data):
    base = Path(project_root) / output_subpath / mode / data
    if not base.exists():
        raise FileNotFoundError(f"Output base not found: {base}")
    # find deepest directories that contain best_flow_model.pth or best_flow_model.pth
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if (d / 'best_flow_model.pth').exists() or (d / 'final_flow_model.pth').exists() or (d / 'best_cnn_model.pth').exists():
            candidates.append(d)
    if not candidates:
        # fallback: search recursively one level deeper
        for d in base.rglob('*'):
            if d.is_dir() and ((d / 'best_flow_model.pth').exists() or (d / 'final_flow_model.pth').exists() or (d / 'best_cnn_model.pth').exists()):
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No trained model directories found under {base}")
    # pick newest modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_flow_wrapper(n_dim, cond_dims, n_blocks=10, n_nodes=512, subnet_depth=3, act='relu'):
    params = {
        'flow': {
            'n_dim': n_dim,
            'n_blocks': n_blocks,
            'n_nodes': n_nodes,
            'cond_dims': cond_dims,
            'subnet_depth': subnet_depth,
            'act': act,
            'load': False,
            'model_location': None,
        }
    }
    return ConditionalInvertibleBlock(params)


def main():
    time_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--mode', choices=['ps2d','ps1d','cnn','skatr'], default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, help='Path to model folder or checkpoint')
    parser.add_argument('--num_sims', type=int, default=None)
    parser.add_argument('--num_samples_per_sim', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--test_indices_path', type=str, default=None, help='Optional path to saved test_indices.npy')
    args = parser.parse_args()

    # load config
    cfg = {}
    if args.config:
        with open(args.config, 'r') as fh:
            cfg = yaml.safe_load(fh) or {}
    
    # merge CLI over YAML
    mode = args.mode or cfg.get('mode', 'ps2d')
    data = args.data or cfg.get('data', 'opt_noise')
    inference = cfg.get('inference', 'flow')
    data_type = cfg.get('data_type', ('skatr' if mode == 'skatr' else 'standard'))
    learn_target = cfg.get('learn_target', 'xhi')
    
    # Extract paths
    project_root = cfg.get('project_root', '/pfs/10/work/hd_pt254-skatr/EoRFlow-dev')
    output_subpath = cfg.get('output_subpath', 'EoRFlow-dev/output')
    
    # Extract model paths
    model_arg = args.model or cfg.get('model_path', None)
    cnn_model_arg = cfg.get('cnn_model_path', None)
    
    # Extract sampling settings
    num_samples_per_sim = args.num_samples_per_sim or cfg.get('num_samples_per_sim', 1000)
    num_sims = args.num_sims or cfg.get('num_sims', None)
    batch_size = cfg.get('batch_size', 16)
    seed = cfg.get('seed', 42)
    test_indices_path = args.test_indices_path or cfg.get('test_indices_path', None)
   
    db_subpath = cfg.get('database_subpath', 'database')
    num_files = cfg.get('num_files', None)
    device = args.device or cfg.get('device', None)
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # redshift dims (match train config)
    min_redshift_index = int(cfg.get('min_redshift_index', 0))
    max_redshift_index = int(cfg.get('max_redshift_index', 15))
    redshift_dim = max_redshift_index - min_redshift_index
    target_dim = redshift_dim

    # compute default cond_dims for flow wrapper
    cond_dims = None
    if mode == 'cnn':
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
        raise ValueError(f"Unknown mode: {mode}")

    # target/logit behavior depends on representation/target type
    if data_type == 'skatr' or mode == 'skatr':
        target_uses_logit = (learn_target == 'xhi')
    else:
        target_uses_logit = True

    # find model dir
    if model_arg:
        model_path = Path(model_arg)
        if model_path.is_file():
            model_dir = model_path.parent
        else:
            model_dir = model_path
    else:
        model_dir = find_latest_model_dir(project_root, output_subpath, mode, data)

    # For CNN mode, look in subdirectories (finetune > flow_train > root)
    # For other modes, look in root directory
    search_paths = []
    if mode == 'cnn':
        # prioritize finetune, then flow_train, then cnn_train, then root
        search_paths = [
            model_dir / 'finetune',
            model_dir / 'flow_train',
            model_dir / 'cnn_train',
            model_dir
        ]
    else:
        search_paths = [model_dir]

    # prefer cfm checkpoints if present, then flow, then joint
    flow_ckpt = None
    detected_backend = None
    
    for search_dir in search_paths:
        if not search_dir.exists():
            continue
            
        best_cfm = search_dir / 'best_cfm_model.pth'
        final_cfm = search_dir / 'final_cfm_model.pth'
        best_flow = search_dir / 'best_flow_model.pth'
        final_flow = search_dir / 'final_flow_model.pth'
        best_joint = search_dir / 'best_joint.pth'

        if best_joint.exists():
            # prioritize joint checkpoint for CNN mode
            ck = torch.load(str(best_joint), map_location='cpu')
            if isinstance(ck, dict) and 'inference_state_dict' in ck:
                # saved CFM-style checkpoint
                flow_ckpt = best_joint; detected_backend = 'cfm'
            elif isinstance(ck, dict) and 'flow' in ck:
                # joint checkpoint that contains a 'flow' state_dict
                flow_ckpt = best_joint; detected_backend = 'joint'
            else:
                # fallback: assume it's a raw flow state_dict
                flow_ckpt = best_joint; detected_backend = 'flow'
            break
        elif best_cfm.exists():
            flow_ckpt = best_cfm; detected_backend = 'cfm'
            break
        elif final_cfm.exists():
            flow_ckpt = final_cfm; detected_backend = 'cfm'
            break
        elif best_flow.exists():
            flow_ckpt = best_flow; detected_backend = 'flow'
            break
        elif final_flow.exists():
            flow_ckpt = final_flow; detected_backend = 'flow'
            break

    # CNN model (for embedding or cnn-mode flow)
    # Search for CNN checkpoint in cnn_train subdirectory first, then root
    best_cnn = None
    if mode == 'cnn':
        cnn_search_paths = [model_dir / 'cnn_train', model_dir / 'finetune', model_dir]
        for cnn_dir in cnn_search_paths:
            if cnn_dir.exists() and (cnn_dir / 'best_cnn_model.pth').exists():
                best_cnn = cnn_dir / 'best_cnn_model.pth'
                break
    else:
        best_cnn = model_dir / 'best_cnn_model.pth' if (model_dir / 'best_cnn_model.pth').exists() else None

    # Prepare dataset to infer dimensions for SKATR mode if needed.
    skcfg = cfg.get('skatr', {})
    dataset_data_dirs = None
    target_norm_stats_path = None
    cond_norm_stats_path = None
    if data_type == 'skatr' or mode == 'skatr':
        source = skcfg.get('source', data)
        source_dirs = skcfg.get('source_dirs', {})
        dataset_data_dirs = skcfg.get('data_dirs', source_dirs.get(source, []))
        if isinstance(dataset_data_dirs, str):
            dataset_data_dirs = [dataset_data_dirs]
        if not dataset_data_dirs:
            raise ValueError("No SKATR data directories configured. Set skatr.data_dirs or skatr.source_dirs[skatr.source].")

        if learn_target == 'sim_params':
            target_norm_stats_path = skcfg.get('target_norm_stats_path', None)
            if target_norm_stats_path is None:
                auto_norm = model_dir / 'target_norm_stats.npz'
                if auto_norm.exists():
                    target_norm_stats_path = str(auto_norm)

        cond_norm_stats_path = skcfg.get('cond_norm_stats_path', None)
        if cond_norm_stats_path is None:
            auto_cnorm = model_dir / 'cond_norm_stats.npz'
            if auto_cnorm.exists():
                cond_norm_stats_path = str(auto_cnorm)

        ds_probe = SkatrGridDataset(
            data_dirs=dataset_data_dirs,
            target=learn_target,
            logit=target_uses_logit,
            min_redshift_index=min_redshift_index,
            max_redshift_index=max_redshift_index,
            sim_param_indices=skcfg.get('sim_param_indices', None),
            drop_tvir=skcfg.get('drop_tvir', True),
            num_sim_params=skcfg.get('num_sim_params', 5),
            normalize_cond=skcfg.get('normalize_cond', False),
            cond_normalization=skcfg.get('cond_normalization', None),
            cond_norm_stats_path=cond_norm_stats_path,
            target_norm_stats_path=target_norm_stats_path,
            xhi_labels_dirs=skcfg.get('xhi_labels_dirs', None),
            xhi_label_key=skcfg.get('xhi_label_key', 'xHI_labels'),
        )
        c0, y0 = ds_probe[0]
        cond_dims = int(c0.numel())
        target_dim = int(y0.numel())
        print(f"Inferred SKATR dims -> cond_dims={cond_dims}, target_dim={target_dim}, target={learn_target}")

    # build appropriate inference model and load state
    model = None
    if detected_backend == 'cfm':
        train_cfg_path = model_dir / 'train_config.yaml'
        train_cfm_cfg = {}
        if train_cfg_path.exists():
            try:
                import yaml as _yaml_load
                with open(train_cfg_path) as _f:
                    _tc = _yaml_load.safe_load(_f) or {}
                train_cfm_cfg = ((_tc.get('fixed_summary') or {}).get('cfm') or {})
                print(f"Auto-loaded CFM hyperparameters from {train_cfg_path}: "
                      f"n_layers={train_cfm_cfg.get('n_layers')}, "
                      f"hidden_dim={train_cfm_cfg.get('hidden_dim')}")
            except Exception as _e:
                print(f"[warn] failed to parse {train_cfg_path}: {_e}; "
                      f"falling back to eval-config / defaults.")
        # Eval-config blocks (either path) used as override / fallback
        eval_fs_cfm = (cfg.get('fixed_summary') or {}).get('cfm') or {}
        eval_short_cfm = cfg.get('cfm') or {}

        def _resolve(key, default):
            for src in (train_cfm_cfg, eval_fs_cfm, eval_short_cfm):
                if src.get(key) is not None:
                    return src[key]
            return default

        cfm_n_layers = int(_resolve('n_layers', 3))
        cfm_hidden   = int(_resolve('hidden_dim', 512))
        cfm_alpha    = float(_resolve('alpha', 0.0))
        cfm_pdrop    = float(_resolve('p_drop', 0.0))
        print(f"CFM architecture: n_layers={cfm_n_layers}, hidden_dim={cfm_hidden}, "
              f"alpha={cfm_alpha}, p_drop={cfm_pdrop}")

        model = ConditionalCFM(n_dim=target_dim, summary_dim=cond_dims, n_layers=cfm_n_layers, hidden_dim=cfm_hidden, alpha=cfm_alpha, p_drop=cfm_pdrop)
        model.to(device)
        if flow_ckpt:
            ck = torch.load(str(flow_ckpt), map_location=device)
            # checkpoint might be wrapped under 'inference_state_dict' or be the raw state_dict
            if isinstance(ck, dict) and 'inference_state_dict' in ck:
                model.load_state_dict(ck['inference_state_dict'])
            elif isinstance(ck, dict) and 'flow' in ck and isinstance(ck['flow'], dict):
                # joint checkpoint where 'flow' may hold the state dict
                model.load_state_dict(ck['flow'])
            else:
                model.load_state_dict(ck)

    else:
        # default to FrEIA flow wrapper
        model = build_flow_wrapper(target_dim, cond_dims)
        model.to(device)
        # the wrapper has a load_model helper
        if flow_ckpt:
            # wrapper.load_model may accept a path to a raw flow state_dict.
            # Some checkpoints are joint containers (e.g. {'cnn':..., 'flow':...}).
            # Try wrapper.load_model first; if it fails, attempt to extract the
            # 'flow' sub-dictionary and load via load_state_dict.
            try:
                loaded = model.load_model(str(flow_ckpt))
            except Exception:
                loaded = False

            if not loaded:
                # fallback: try loading via state_dict directly
                try:
                    sd = torch.load(str(flow_ckpt), map_location=device)
                    # if joint checkpoint, extract the 'flow' sub-state
                    if isinstance(sd, dict) and 'flow' in sd and isinstance(sd['flow'], dict):
                        flow_state = sd['flow']
                    elif isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                        flow_state = sd['state_dict']
                    else:
                        # assume sd itself is the flow state_dict
                        flow_state = sd

                    # if the wrapper exposes load_state_dict, use it; otherwise try loader helper
                    try:
                        model.load_state_dict(flow_state)
                    except Exception as e:
                        # final attempt: save flow_state temporarily and call wrapper.load_model
                        tmp = Path('/tmp') / f'flow_state_{int(time.time())}.pth'
                        torch.save(flow_state, str(tmp))
                        loaded = model.load_model(str(tmp))
                        try:
                            tmp.unlink()
                        except Exception:
                            pass
                        if not loaded:
                            raise e
                except Exception as e:
                    raise FileNotFoundError(f"Failed to load flow checkpoint: {flow_ckpt} -> {e}")

    # prepare dataset to draw conditions
    data_dirs = [os.path.join(project_root, db_subpath, data, f'batch_0')]
    print(f"Using model dir: {model_dir}, flow_ckpt: {flow_ckpt}")
    if data_type == 'skatr' or mode == 'skatr':
        print(f"Preparing SKATR dataset from: {dataset_data_dirs}")
        ds = SkatrGridDataset(
            data_dirs=dataset_data_dirs,
            target=learn_target,
            logit=target_uses_logit,
            min_redshift_index=min_redshift_index,
            max_redshift_index=max_redshift_index,
            sim_param_indices=skcfg.get('sim_param_indices', None),
            drop_tvir=skcfg.get('drop_tvir', True),
            num_sim_params=skcfg.get('num_sim_params', 5),
            normalize_cond=skcfg.get('normalize_cond', False),
            cond_normalization=skcfg.get('cond_normalization', None),
            cond_norm_stats_path=cond_norm_stats_path,
            target_norm_stats_path=target_norm_stats_path,
            xhi_labels_dirs=skcfg.get('xhi_labels_dirs', None),
            xhi_label_key=skcfg.get('xhi_label_key', 'xHI_labels'),
        )
    else:
        print(f"Preparing standard dataset from: {data_dirs}")
        ds = EoRH5Dataset(
            data_dirs=data_dirs,
            mode=mode,
            min_redshift_index=min_redshift_index,
            max_redshift_index=max_redshift_index,
            logit=True,
            num_files=num_files,
        )

    total = len(ds)
    if total == 0:
        raise RuntimeError('Dataset appears empty; check data_dirs and num_files')

    rng = np.random.default_rng(seed)
    # If test indices are provided (or auto-discovered), restrict evaluation to held-out set.
    if test_indices_path is None:
        auto_test_indices = model_dir / 'test_indices.npy'
        if auto_test_indices.exists():
            test_indices_path = str(auto_test_indices)

    if test_indices_path:
        raw_test_idxs = np.load(test_indices_path)
        raw_test_idxs = np.asarray(raw_test_idxs, dtype=np.int64).reshape(-1)
        # Guard against accidental out-of-range values.
        available_idxs = raw_test_idxs[(raw_test_idxs >= 0) & (raw_test_idxs < total)]
        if available_idxs.size == 0:
            raise RuntimeError(f'No valid test indices found in {test_indices_path} for dataset size {total}')
        if num_sims is None:
            num_sims = int(available_idxs.size)
        num_sims = min(int(num_sims), int(available_idxs.size))
        sim_idxs = rng.choice(available_idxs, size=num_sims, replace=False)
        print(f"Using held-out test indices from {test_indices_path}: {num_sims}/{available_idxs.size} sims")
    else:
        if num_sims is None:
            num_sims = total
        num_sims = min(int(num_sims), total)
        sim_idxs = rng.choice(total, size=num_sims, replace=False)
        print(f"Using random simulation indices from full dataset: {num_sims}/{total} sims")

    n_dim = target_dim
    samples_per_sim = int(num_samples_per_sim)

    samples_batches = []
    conds_batches = []
    labels_batches = []

    # If mode == 'cnn' and we don't have a flow loaded, assume flow_ckpt may be in same dir; else we may need to generate embeddings using CNN
    # If cnn model exists we can embed dataset
    cnn_model = None
    if mode == 'cnn':
        cnn_model = ConvNet3D({'cnn': {}}, in_ch=1, N_parameter=redshift_dim)
        cnn_model.to(device)
        if best_cnn is not None and best_cnn.exists():
            cnn_model.load_state_dict(torch.load(best_cnn, map_location=device))
        elif detected_backend == 'joint' and flow_ckpt is not None:
            # try to load CNN from joint checkpoint
            try:
                ck = torch.load(str(flow_ckpt), map_location=device)
                if isinstance(ck, dict) and 'cnn' in ck:
                    cnn_model.load_state_dict(ck['cnn'])
            except Exception as e:
                print(f"Warning: Could not load CNN from joint checkpoint: {e}")

    # sampling loop by batches of sims
    n_batches = (num_sims + batch_size - 1) // batch_size
    for ib in range(n_batches):
        start = ib * batch_size
        end = min(start + batch_size, num_sims)
        cur_idxs = sim_idxs[start:end]
        batch_sims = len(cur_idxs)

        cur_conds = []
        cur_labels = []
        for ii in cur_idxs:
            c, y = ds[ii]
            # if cnn-mode and we have a cnn_model, compute embedding
            if mode == 'cnn' and cnn_model is not None:
                with torch.no_grad():
                    cnn_model.eval()
                    c_in = torch.tensor(c, dtype=torch.float32).to(device)
                    emb = cnn_model(c_in.unsqueeze(0).unsqueeze(1)).cpu().numpy()[0]
                    cur_conds.append(emb)
            else:
                cur_conds.append(c.numpy())
            cur_labels.append(y.numpy())

        c_arr = np.stack(cur_conds, axis=0)     # (batch_sims, cond_dim)
        label_arr = np.stack(cur_labels, axis=0) # (batch_sims, n_dim)

        # expand conditions and sample
        c = torch.tensor(c_arr, dtype=torch.float32).to(device)
        # make conditioning tensor shape (total_draws, cond_dim) or (cond_dim,) depending on model.sample
        total_draws = batch_sims * samples_per_sim
        c_exp = c.repeat_interleave(samples_per_sim, dim=0)
        with torch.no_grad():
            # unified sampling API: both ConditionalCFM and ConditionalInvertibleBlock implement sample(n_samples, cond)
            x = model.sample(total_draws, c_exp)
            # sample may return tensor of shape (total_draws, n_dim)
            if isinstance(x, tuple) or (hasattr(x, '__len__') and len(x) == 2 and torch.is_tensor(x[0])):
                # backward compatibility: some wrappers returned (x, _)
                x = x[0]

        # The flow returns a flat array ordered by simulation-blocks (all samples
        # for sim0, then all for sim1, ...). To get shape (samples_per_sim,
        # batch_sims, n_dim) we first reshape to (batch_sims, samples_per_sim,
        # n_dim) and then transpose axes so samples become the first axis.
        x_np = x.cpu().numpy().reshape(batch_sims, samples_per_sim, n_dim).transpose(1, 0, 2)
        samples_batches.append(x_np)
        conds_batches.append(c_arr)
        labels_batches.append(label_arr)

    samples_all = np.concatenate(samples_batches, axis=1) 
    conds_all = np.concatenate(conds_batches, axis=0)
    labels_all = np.concatenate(labels_batches, axis=0)

    if target_uses_logit:
        # Convert from logit-space to [0, 1] only for xHI targets.
        preds_out = 1.0 / (1.0 + np.exp(-samples_all))
        labels_out = 1.0 / (1.0 + np.exp(-labels_all))
    elif (data_type == 'skatr' or mode == 'skatr') and learn_target == 'sim_params' and getattr(ds, 'target_norm_stats', None) is not None:
        # Convert back to physical parameter units for reporting/plotting.
        preds_out = ds.denormalize_targets(samples_all)
        labels_out = ds.denormalize_targets(labels_all)
    else:
        preds_out = samples_all
        labels_out = labels_all

    # Posterior-volume summary for plotting/reporting.
    # Definition follows relative uncertainty metric:
    # mean posterior std across observations, normalized by std(labels) on test set.
    posterior_volume = None
    sample_std = np.std(preds_out, axis=0)  # (num_obs, n_params)
    mean_posterior_std = np.mean(sample_std, axis=0)
    label_std = np.std(labels_out, axis=0)
    rel_unc = mean_posterior_std / (label_std + 1e-10)
    posterior_volume = dict(
        definition='mean_posterior_std_over_label_std',
        per_param_relative_uncertainty=rel_unc.tolist(),
        mean_posterior_std=mean_posterior_std.tolist(),
        label_std=label_std.tolist(),
        volume_fraction_relative_uncertainty=float(np.prod(rel_unc)),
    )

    out_file = Path(model_dir) / f'samples.npz'
    np.savez_compressed(
        out_file,
        preds=preds_out,
        labels=labels_out,
        conds=conds_all,
        info=dict(
            mode=mode,
            data=data,
            data_type=data_type,
            learn_target=learn_target,
            target_uses_logit=target_uses_logit,
            cond_norm_stats_path=cond_norm_stats_path,
            target_norm_stats_path=target_norm_stats_path,
            test_indices_path=test_indices_path,
            n_dim=n_dim,
            cond_dims=cond_dims,
            num_sims=num_sims,
            samples_per_sim=samples_per_sim,
            model_dir=str(model_dir),
            posterior_volume=posterior_volume,
        )
    )

    print(f"Saved samples with shape (samples_per_sim, num_sims, dim) = {preds_out.shape} to: {out_file}")
    time_end = time.time()
    print(f"Total sampling time: {time_end - time_start:.1f} sec")


if __name__ == '__main__':
    main()

