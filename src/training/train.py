#!/usr/bin/env python
import os
import sys
import logging
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import time
import shutil

# Add custom paths
sys.path.append('/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/src/models')
sys.path.append('/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/src/data_tools')  # For Loreli loader

from flow import ConditionalInvertibleBlock
from cfm import ConditionalCFM
from data_loader import EoRH5Dataset, SkatrGridDataset
from loreli_data_loader import LoreliPS1DDataset
from cnn import ConvNet3D

# training script for EoRFlow
# includes ps1d, ps2d, cnn, and skatr modes

# ---------------------- CLI ----------------------
parser = argparse.ArgumentParser(description='Train model for EoRFlow')
parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
parser.add_argument('--mode', type=str, choices=['ps2d','ps1d','cnn', 'skatr'], default=None)
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--inference', type=str, choices=['flow','cfm'], default=None, help='Which inference backend to use (flow or cfm)')
parser.add_argument('--data_type', type=str, choices=['standard', 'skatr'], default=None, help='Input representation type')
parser.add_argument('--learn_target', type=str, choices=['xhi', 'sim_params'], default=None, help='Learning target for SKATR data')
args = parser.parse_args()

mode = args.mode
data = args.data

# ------------------- LOAD CONFIG (optional) -------------------
config = {}
if args.config:
    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh) or {}

# allow config file to supply defaults, CLI args override if provided
mode = mode or config.get('mode', 'ps2d')
data = data or config.get('data', 'opt_noise')
data_type = args.data_type or config.get('data_type', ('skatr' if mode == 'skatr' else 'standard'))
learn_target = args.learn_target or config.get('learn_target', 'xhi')

# ------------------- HYPERPARAMS -------------------
# Global settings
project_root = config.get('project_root', '/pfs/10/work/hd_pt254-eorflow')
min_redshift_index = config.get('min_redshift_index', 0)
max_redshift_index = config.get('max_redshift_index', 15)
redshift_dim = max_redshift_index - min_redshift_index
target_dim = redshift_dim
cond_dims = None

# Determine which config section to use
if mode in ['ps1d', 'ps2d', 'skatr']:
    # Fixed summary modes
    training_cfg = config.get('fixed_summary', {})
    lr = training_cfg.get('lr', 1e-3)
    batch_size = config.get('batch_size', 16)
    num_epochs = training_cfg.get('epochs', 1000)
    weight_decay = training_cfg.get('weight_decay', 1e-4)
    patience = training_cfg.get('patience', 10)
    
    # Flow/CFM config
    flow_cfg = training_cfg.get('flow', {})
    cfm_cfg = training_cfg.get('cfm', {})
    
    # Flow architecture params
    n_blocks = flow_cfg.get('n_blocks', 10)
    n_nodes = flow_cfg.get('n_nodes', 512)
    subnet_depth = flow_cfg.get('subnet_depth', 3)
    
elif mode == 'cnn':
    # CNN multi-stage training
    cnn_multistage = config.get('cnn_multistage', {})
    stage = cnn_multistage.get('stage', 'cnn')
    batch_size = config.get('batch_size', 16)
    
    # CNN pretrain params
    cnn_cfg = cnn_multistage.get('cnn_pretrain', {})
    lr_cnn = cnn_cfg.get('lr', 4e-4)
    wd_cnn = cnn_cfg.get('weight_decay', 1e-5)
    cnn_epochs = cnn_cfg.get('epochs', 200)
    
    # Inference network params
    inference_cfg = cnn_multistage.get('inference_train', {})
    lr_flow = inference_cfg.get('lr', 1e-3)
    wd_flow = inference_cfg.get('weight_decay', 1e-4)
    flow_epochs = inference_cfg.get('epochs', 200)
    cfm_epochs = flow_epochs  # Same default
    
    flow_cfg = inference_cfg.get('flow', {})
    cfm_cfg = inference_cfg.get('cfm', {})
    
    # Flow architecture params
    n_blocks = flow_cfg.get('n_blocks', 10)
    n_nodes = flow_cfg.get('n_nodes', 512)
    subnet_depth = flow_cfg.get('subnet_depth', 3)
    
    # Finetune params
    finetune_cfg = cnn_multistage.get('finetune', {})
    finetune_epochs = finetune_cfg.get('epochs', 200)
    flow_warmup_epochs = finetune_cfg.get('warmup_epochs', 5)
    finetune_lr = finetune_cfg.get('lr', 1e-4)
    wd_cnn_ft = finetune_cfg.get('weight_decay_cnn', 1e-5)
    wd_inf_ft = finetune_cfg.get('weight_decay_inference', 1e-4)
    
    # Set defaults for use in flow training
    lr = lr_flow
    weight_decay = wd_flow
    num_epochs = 1000  # Not used in CNN mode
    
else:
    raise ValueError(f"Unknown mode: {mode}")

# Generate output tag
out_tag = config.get('output_tag') or f'{n_blocks}_{n_nodes}'
output_dir = os.path.join(project_root, config.get('output_subpath', 'EoRFlow-dev/output'), mode, data, out_tag)
os.makedirs(output_dir, exist_ok=True)

# dump the full training config to the output dir
try:
    import yaml as _yaml_dump
    with open(os.path.join(output_dir, 'train_config.yaml'), 'w') as _f:
        _yaml_dump.safe_dump(config, _f, sort_keys=False)
except Exception as _e:
    # Don't fail training just because the config dump failed.
    print(f"[warn] could not save train_config.yaml to {output_dir}: {_e}")

# Common settings
add_noise = config.get('add_noise', False)
num_files = config.get('num_files', None)
use_amp_config = bool(config.get('use_amp', False))

# Get inference backend
inference = args.inference or config.get('inference', 'flow')

# ------------------- STAGE FOLDERS & LOGGING -------------------
cnn_stage_dir = os.path.join(output_dir, 'cnn_train')
flow_stage_dir = os.path.join(output_dir, 'flow_train')
finetune_stage_dir = os.path.join(output_dir, 'finetune')
if mode == 'cnn':
    os.makedirs(cnn_stage_dir, exist_ok=True)
    os.makedirs(flow_stage_dir, exist_ok=True)
    os.makedirs(finetune_stage_dir, exist_ok=True)
    # choose the active stage dir based on configured stage
    if stage == 'cnn':
        active_stage_dir = cnn_stage_dir
    elif stage == 'flow':
        active_stage_dir = flow_stage_dir
    elif stage == 'finetune':
        active_stage_dir = finetune_stage_dir
    else:
        active_stage_dir = output_dir
    log_path = os.path.join(active_stage_dir, 'train.log')
else:
    # non-cnn modes write to root output_dir
    log_path = os.path.join(output_dir, 'train.log')

# create logging
logging.basicConfig(
    filename=log_path,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.info(f"Logging to {log_path}")
logging.info(f"Starting training in mode={mode}, data={data}")
logging.info(f"data_type={data_type}, learn_target={learn_target}")

# ------------------- DEVICE -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
# enable cuDNN autotuner for potentially faster kernels when input sizes are constant
try:
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logging.info('Enabled torch.backends.cudnn.benchmark = True')
except Exception:
    pass

# ------------------- DATASET -------------------
if data_type == 'skatr' or mode == 'skatr':
    skcfg = config.get('skatr', {})
    source = skcfg.get('source', data)
    source_dirs = skcfg.get('source_dirs', {})
    data_dirs = skcfg.get('data_dirs', source_dirs.get(source, []))
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    # cfm_aug: when True, sample one of n_aug slots per __getitem__ for training, and slot 0 (identity) for val/test. 
    # When cfm_aug=False (default) we instantiate a single dataset and alias both.
    cfm_aug = bool(skcfg.get('cfm_aug', False))

    _dataset_kwargs = dict(
        data_dirs=data_dirs,
        target=learn_target,
        logit=(learn_target == 'xhi'),
        min_redshift_index=min_redshift_index,
        max_redshift_index=max_redshift_index,
        sim_param_indices=skcfg.get('sim_param_indices', None),
        drop_tvir=skcfg.get('drop_tvir', True),
        num_sim_params=skcfg.get('num_sim_params', 5),
        normalize_cond=skcfg.get('normalize_cond', False),
        cond_normalization=skcfg.get('cond_normalization', None),
        cond_norm_stats_path=skcfg.get('cond_norm_stats_path', None),
        xhi_labels_dirs=skcfg.get('xhi_labels_dirs', None),
        xhi_label_key=skcfg.get('xhi_label_key', 'xHI_labels'),
        num_files=num_files,
    )
    if cfm_aug:
        train_dataset = SkatrGridDataset(**_dataset_kwargs, cfm_aug=True)
        eval_dataset  = SkatrGridDataset(**_dataset_kwargs, cfm_aug=False)
        dataset = eval_dataset  # used below for sample peek, len(), split sanity
        logging.info(
            f"cfm_aug=True: D4 orbit size detected = {train_dataset.n_aug_slots}. "
            "Train sees random slots; val/test see slot 0 (identity)."
        )
    else:
        dataset = SkatrGridDataset(**_dataset_kwargs, cfm_aug=False)
        train_dataset = eval_dataset = dataset

    sample_cond, sample_target = dataset[0]
    cond_dims = int(sample_cond.numel())
    target_dim = int(sample_target.numel())

    logging.info(
        f"Loaded SKATR dataset: {len(dataset)} samples, source={source}, "
        f"cond_dim={cond_dims}, target_dim={target_dim}, target={learn_target}"
    )

elif data == 'loreli':
    # Loreli II dataset - specialized ps1d loader
    loreli_root = config.get('loreli_root', '/pfs/10/work/hd_pt254-skatr/Loreli_II')
    dataset = LoreliPS1DDataset(
        loreli_root=loreli_root,
        min_redshift_index=min_redshift_index,
        max_redshift_index=max_redshift_index,
        logit=True,
        add_noise=add_noise,
        num_samples=num_files,
        verbose=True,
    )
    # Override redshift_dim for Loreli (has 12 usable redshifts, not 15)
    redshift_dim = dataset.n_redshifts
    logging.info(f"Loaded Loreli dataset: {len(dataset)} samples, {redshift_dim} redshifts")
    logging.info(f"Redshift range: z=[{dataset.ps_redshifts.min():.2f}, {dataset.ps_redshifts.max():.2f}]")
else:
    # Original EoRFlow datasets (pure, opt_noise, etc.)
    database_path = config.get('database_path', 'database')
    data_dirs = [os.path.join(database_path, data, f'batch_{i}') for i in range(0, 4)]  # Include batch_0
    
    if num_files is None:
        total_files = 0
        for d in data_dirs:
            if os.path.exists(d):
                files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
                total_files += len(files)
    else:
        total_files = num_files
        
    # Ensure we have at least batch_size files
    num_files = max(batch_size, (total_files // batch_size) * batch_size)
    logging.info(f"Total files found: {total_files}, using {num_files} files for training/validation.")
    
    dataset = EoRH5Dataset(
        data_dirs=data_dirs,
        mode=mode,
        min_redshift_index=min_redshift_index,
        max_redshift_index=max_redshift_index,
        logit=True,
        add_noise=add_noise,
        num_files=num_files
    )
    logging.info(f"Total samples available: {len(dataset)}")
    logging.info(f"Using redshift indices [{min_redshift_index}, {max_redshift_index}) -> {max_redshift_index - min_redshift_index} redshifts")

# Ensure train_dataset / eval_dataset are defined for every data branch.
try:
    train_dataset
except NameError:
    train_dataset = dataset
try:
    eval_dataset
except NameError:
    eval_dataset = dataset

# split train/val/test
test_fraction = config.get('test_fraction', 0.0)  # Default no test set
val_fraction = config.get('val_fraction', 0.2)    # Default 20% val from remaining
split_indices_path = config.get('split_indices_path', None)

# Sanity check: cfm_aug requires the canonical-split path because we rely on distinct train_dataset/eval_dataset split
if train_dataset is not eval_dataset and split_indices_path is None:
    raise RuntimeError(
        "cfm_aug=True requires `split_indices_path` (canonical-split mode) so train "
        "and val/test can be wrapped around their respective dataset views. "
        "Either set split_indices_path or disable cfm_aug."
    )

if split_indices_path is not None:
    logging.info(f"Loading split indices from {split_indices_path}")
    split = np.load(split_indices_path)
    train_idx = split['train'].astype(np.int64).tolist()
    val_idx   = split['val'].astype(np.int64).tolist()
    test_idx  = split['test'].astype(np.int64).tolist()
    expected_total = int(split['total'].item()) if 'total' in split.files else len(dataset)
    if expected_total != len(dataset):
        raise RuntimeError(
            f"split_indices_path total ({expected_total}) does not match dataset size "
            f"({len(dataset)}). Refusing to proceed silently."
        )
    # cfm_aug case: train_dataset has cfm_aug=True (random slot per __getitem__)
    # eval_dataset has cfm_aug=False (slot 0 = identity). 
    train_ds = torch.utils.data.Subset(train_dataset, train_idx)
    val_ds   = torch.utils.data.Subset(eval_dataset, val_idx)
    test_ds  = torch.utils.data.Subset(eval_dataset, test_idx)
    train_n, val_n, test_n = len(train_ds), len(val_ds), len(test_ds)
    logging.info(f"Dataset split (from file): train={train_n}, val={val_n}, test={test_n}")
    np.save(os.path.join(output_dir, 'test_indices.npy'), np.asarray(test_idx, dtype=np.int64))
    logging.info(f"Saved {test_n} test indices to {output_dir}/test_indices.npy")

elif test_fraction > 0:
    # Three-way split: test, then train/val from remainder
    test_n = int(test_fraction * len(dataset))
    remainder_n = len(dataset) - test_n
    val_n = int(val_fraction * remainder_n)
    train_n = remainder_n - val_n

    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n],
                                               generator=torch.Generator().manual_seed(42))
    logging.info(f"Dataset split: train={train_n}, val={val_n}, test={test_n}")

    # Save test indices for later evaluation
    test_indices = test_ds.indices
    np.save(os.path.join(output_dir, 'test_indices.npy'), np.array(test_indices))
    logging.info(f"Saved {len(test_indices)} test indices to {output_dir}/test_indices.npy")
else:
    # Original two-way split
    train_n = int(0.8 * len(dataset))
    val_n   = len(dataset) - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])
    logging.info(f"Dataset split: train={train_n}, val={val_n}")

# Optional train-split target normalization for SKATR sim-parameter regression
if (data_type == 'skatr' or mode == 'skatr') and learn_target == 'sim_params':
    skcfg = config.get('skatr', {})
    norm_enabled = bool(skcfg.get('normalize_sim_params', False))
    if norm_enabled:
        train_indices = getattr(train_ds, 'indices', None)
        if train_indices is None:
            train_indices = range(len(dataset))

        stats = dataset.fit_target_normalization(indices=train_indices)
        stats_path = os.path.join(output_dir, 'target_norm_stats.npz')
        dataset.save_target_normalization_stats(stats_path)
        logging.info(
            f"Fitted SKATR target normalization on train split and saved stats to {stats_path}. "
            f"log10_indices={stats.get('log10_indices', [])}"
        )

# Optional train-split condition normalization for SKATR embeddings
if data_type == 'skatr' or mode == 'skatr':
    skcfg = config.get('skatr', {})
    cond_norm_mode = skcfg.get('cond_normalization', None)
    if cond_norm_mode is None and bool(skcfg.get('normalize_cond', False)):
        cond_norm_mode = 'per_sample_minmax'

    if cond_norm_mode == 'zscore':
        train_indices = getattr(train_ds, 'indices', None)
        if train_indices is None:
            train_indices = range(len(dataset))

        cstats = dataset.fit_cond_normalization(indices=train_indices)
        cstats_path = os.path.join(output_dir, 'cond_norm_stats.npz')
        dataset.save_cond_normalization_stats(cstats_path)
        logging.info(
            f"Fitted SKATR condition normalization on train split and saved stats to {cstats_path}."
        )

if (data_type == 'skatr' or mode == 'skatr') and train_dataset is not dataset:
    if getattr(dataset, 'target_norm_stats', None) is not None:
        train_dataset.set_target_normalization_stats(dataset.target_norm_stats)
        logging.info("Propagated target_norm_stats from eval_dataset to train_dataset (cfm_aug).")
    if getattr(dataset, 'cond_norm_stats', None) is not None:
        train_dataset.set_cond_normalization_stats(dataset.cond_norm_stats)
        logging.info("Propagated cond_norm_stats from eval_dataset to train_dataset (cfm_aug).")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.get('num_workers', 2), pin_memory=config.get('pin_memory', True), persistent_workers=config.get('persistent_workers', True))
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=config.get('num_workers', 2), pin_memory=config.get('pin_memory', True), persistent_workers=config.get('persistent_workers', True))
print(f'batch size: {batch_size}, num workers: {config.get("num_workers", 2)}, pin memory: {config.get("pin_memory", True)}')
    


if mode == 'cnn':
    # instantiate the CNN summary network
    cnn_model = ConvNet3D({'cnn': {}}, in_ch=1, N_parameter=redshift_dim)
    cnn_model.to(device)
    cond_dims = redshift_dim

elif mode == 'ps2d':
    # flattened 2D PS per slice + one z per slice
    obs_dim = redshift_dim * 10 * 10
    cond_dims = obs_dim + redshift_dim

elif mode == 'ps1d':
    # flattened 1D PS per slice + one z per slice
    obs_dim = redshift_dim * 14
    cond_dims = obs_dim + redshift_dim

elif mode == 'skatr':
    if data_type != 'skatr':
        raise ValueError("mode='skatr' requires data_type='skatr'")
    if cond_dims is None:
        raise ValueError("SKATR cond_dims were not inferred. Check skatr.data_dirs configuration.")

else:
    raise ValueError(f"Unknown mode: {mode}")

# set up inference model (flow or cfm)
model_params = {
    'flow': {
        'n_dim': target_dim,
        'n_blocks': n_blocks,
        'n_nodes': n_nodes,
        'cond_dims': cond_dims,            
        'subnet_depth': subnet_depth,                       
        'act': flow_cfg.get('act', 'relu'),                  
        'load': False,
        'model_location': None,
    }
}
inference = args.inference or config.get('inference', 'flow')
if inference == 'flow':
    flow_model = ConditionalInvertibleBlock(model_params)
    flow_model.to(device)
elif inference == 'cfm':
    # cfm_cfg was already loaded from the appropriate section (line 69 or 96)
    n_layers = cfm_cfg.get('n_layers', 3)
    hidden_dim = cfm_cfg.get('hidden_dim', 512)
    alpha = cfm_cfg.get('alpha', 0.0)
    p_drop = cfm_cfg.get('p_drop', 0.0)
    flow_model = ConditionalCFM(n_dim=target_dim, summary_dim=cond_dims, n_layers=n_layers, hidden_dim=hidden_dim, alpha=alpha, p_drop=p_drop)
    flow_model.to(device)
else:
    raise ValueError(f"Unknown inference backend: {inference}")

# helpers available to all training stages so CNN pipeline functions can call them
def model_train(m):
    # prefer setting .flow if present (FrEIA wrapper), otherwise call .train()
    if hasattr(m, 'flow'):
        try:
            m.flow.train()
        except Exception:
            pass
    elif hasattr(m, 'train'):
        m.train()


def model_eval(m):
    if hasattr(m, 'flow'):
        try:
            m.flow.eval()
        except Exception:
            pass
    elif hasattr(m, 'eval'):
        m.eval()


def model_state_dict_for_saving(m, backend_name: str):
    sd = m.state_dict()
    if backend_name == 'cfm':
        return {'inference_state_dict': sd}
    return sd


# ------------------- Diagnostics helpers -------------------
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

def _read_mem_from_proc():
    # returns (used_percent, used_mb, total_mb)
    try:
        with open('/proc/meminfo', 'r') as fh:
            lines = fh.readlines()
        info = {}
        for L in lines:
            if ':' not in L:
                continue
            k, v = L.split(':', 1)
            info[k.strip()] = v.strip()
        mem_total_kb = int(info.get('MemTotal', '0 kB').split()[0])
        mem_free_kb = int(info.get('MemFree', '0 kB').split()[0])
        buffers_kb = int(info.get('Buffers', '0 kB').split()[0])
        cached_kb = int(info.get('Cached', '0 kB').split()[0])
        used_kb = mem_total_kb - mem_free_kb - buffers_kb - cached_kb
        total_mb = mem_total_kb / 1024.0
        used_mb = used_kb / 1024.0
        pct = 100.0 * used_kb / mem_total_kb if mem_total_kb>0 else 0.0
        return pct, used_mb, total_mb
    except Exception:
        return 0.0, 0.0, 0.0

def _get_cpu_mem_usage():
    if _HAS_PSUTIL:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return cpu, mem.percent, mem.used/1024/1024.0, mem.total/1024/1024.0
    else:
        try:
            load1, load5, load15 = os.getloadavg()
            cores = os.cpu_count() or 1
            cpu_pct = min(100.0, 100.0 * load1 / cores)
        except Exception:
            cpu_pct = 0.0
        mem_pct, used_mb, total_mb = _read_mem_from_proc()
        return cpu_pct, mem_pct, used_mb, total_mb

def _get_gpu_stats_torch():
    if not torch.cuda.is_available():
        return None
    try:
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        reserved = torch.cuda.memory_reserved(dev) / (1024**2)
        allocated = torch.cuda.memory_allocated(dev) / (1024**2)
        total = props.total_memory / (1024**2)
        pct_mem = 100.0 * allocated / total if total>0 else 0.0
        return {'mem_used_mb': allocated, 'mem_reserved_mb': reserved, 'mem_total_mb': total, 'mem_pct': pct_mem}
    except Exception:
        return None

def _get_gpu_stats_nvidia_smi():
    # try nvidia-smi (csv, no units)
    try:
        cmd = shutil.which('nvidia-smi')
        if not cmd:
            return None
        out = subprocess.check_output([cmd, '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL)
        s = out.decode('utf8').strip().splitlines()
        # support multiple GPUs (return list)
        gpus = []
        for line in s:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({'util_pct': float(parts[0]), 'mem_util_pct': float(parts[1]), 'mem_used_mb': float(parts[2]), 'mem_total_mb': float(parts[3])})
        return gpus
    except Exception:
        return None

def log_system_stats(prefix=''):
    cpu_pct, mem_pct, used_mb, total_mb = _get_cpu_mem_usage()
    gpu_smi = _get_gpu_stats_nvidia_smi()
    gpu_torch = _get_gpu_stats_torch()
    parts = [f"CPU={cpu_pct:.1f}%", f"RAM={mem_pct:.1f}% ({used_mb:.0f}/{total_mb:.0f}MB)"]
    if gpu_smi:
        for i, g in enumerate(gpu_smi):
            parts.append(f"GPU{i}: util={g['util_pct']:.0f}%, mem={g['mem_util_pct']:.0f}% ({g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f}MB)")
    elif gpu_torch:
        parts.append(f"GPU0: mem={gpu_torch['mem_pct']:.1f}% ({gpu_torch['mem_used_mb']:.0f}/{gpu_torch['mem_total_mb']:.0f}MB)")
    msg = f"{prefix} | " + ' | '.join(parts)
    logging.info(msg)
    return msg

# optimizer & scheduler
optimizer = optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ------------------- LOSS FUNCTION -------------------
def flow_loss(model, y, cond, n_dim):
    # Expects y in logit space (unconstrained real line) as provided by data_loader
    # prefer batch_loss API when present
    if hasattr(model, 'batch_loss'):
        return model.batch_loss(y, cond)
    # fallback: assume model is a SequenceINN-like callable
    # DEBUG: Add shape logging before model call
    if not hasattr(flow_loss, '_debug_printed'):
        print(f"\n=== DEBUG flow_loss ===")
        print(f"Before model call: y.shape={y.shape}, cond.shape={cond.shape}, n_dim={n_dim}")
        print(f"=======================\n")
        flow_loss._debug_printed = True
    
    z, jac = model(y, c=[cond])
    return (0.5 * torch.sum(z**2, dim=1) - jac).mean() / n_dim


# ------------------- CNN / FLOW helpers (cnn mode) -------------------
if mode == 'cnn':
    # import pretrain-style helpers where useful
    mse = torch.nn.MSELoss()

    def train_cnn_full(cnn, train_loader, val_loader, output_dir):
        best_val = float('inf')
        opt = optim.AdamW(cnn.parameters(), lr=lr_cnn, weight_decay=wd_cnn)
        train_losses, val_losses = [], []
        logging.info('CNN pretraining starting. Initial system stats:')
        log_system_stats(prefix='[CNN PRETRAIN START]')

        # AMP setup: explicit config toggle, only active on CUDA
        use_amp = use_amp_config and (device.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        logging.info(f'AMP requested: {use_amp_config}')
        logging.info(f'AMP enabled: {use_amp}')
        if use_amp_config and device.type != 'cuda':
            logging.warning('AMP requested but CUDA is not available; running with AMP disabled.')

        for ep in range(1, cnn_epochs+1):
            logging.info(f'[CNN] Epoch {ep} start')
            # epoch timer
            t_ep_start = time.time()
            cnn.train()
            running = 0.0
            for bidx, (imgs, labels) in enumerate(train_loader, start=1):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                labels = torch.sigmoid(labels)  # map from logit (from data_loader) to [0,1] for CNN training
                opt.zero_grad()
                # forward with autocast when using AMP
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = cnn(imgs.unsqueeze(1))
                        loss = mse(pred, labels)
                    # scale + backward + step
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = cnn(imgs.unsqueeze(1))
                    loss = mse(pred, labels)
                    loss.backward()
                    opt.step()
                running += loss.item()
            train_losses.append(running / len(train_loader))

            cnn.eval()
            running = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    labels = torch.sigmoid(labels)  # map from logit (from data_loader) to [0,1] for CNN validation
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            running += mse(cnn(imgs.unsqueeze(1)), labels).item()
                    else:
                        running += mse(cnn(imgs.unsqueeze(1)), labels).item()
            val_losses.append(running / len(val_loader))

            # epoch timing and one-line diagnostics
            t_ep_end = time.time()
            epoch_time = t_ep_end - t_ep_start
            train_avg = train_losses[-1]
            val_avg = val_losses[-1]
            logging.info(f"[CNN] Ep{ep}: time={(epoch_time/60):.1f}min, train={train_avg:.6f}, val={val_avg:.6f}")
            log_system_stats(prefix=f'[CNN E{ep} STATS]')

            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                torch.save(cnn.state_dict(), os.path.join(output_dir, 'best_cnn_model.pth'))

        # save final
        torch.save(cnn.state_dict(), os.path.join(output_dir, 'final_cnn_model.pth'))
        try:
            plt.plot(train_losses, label='train'); plt.plot(val_losses, label='val'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'cnn_loss.pdf'))
        except Exception:
            logging.exception('Failed to plot CNN losses')
        np.save(os.path.join(output_dir, 'cnn_train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(output_dir, 'cnn_val_losses.npy'), np.array(val_losses))

    def embed_dataset(cnn, full_dataset, output_dir):
        # create embeddings for all data and save as npz
        cnn.eval()
        all_c, all_y = [], []
        loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('num_workers', 2), pin_memory=config.get('pin_memory', True))
        with torch.no_grad():
            for img, y in loader:
                img = img.to(device)
                c = cnn(img.unsqueeze(1)).cpu().numpy()
                all_c.append(c)
                all_y.append(y.numpy())
        cond = np.vstack(all_c)
        lab = np.vstack(all_y)
        np.savez(os.path.join(output_dir, 'embeds.npz'), cond=cond.astype('f4'), y=lab.astype('f4'))
        logging.info("Embeddings saved.")

    class CondDataset(torch.utils.data.Dataset):
        def __init__(self, npz):
            d = np.load(npz)
            self.cond = d['cond']
            self.y = d['y']
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.cond[idx], self.y[idx]

    def train_flow_from_embeds(flow, embed_npz, output_dir):
        ds = CondDataset(embed_npz)
        n = len(ds)
        train_n = int(0.8 * n)
        train_ds, val_ds = random_split(ds, [train_n, n - train_n], generator=torch.Generator().manual_seed(42))
        tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.get('num_workers', 2), pin_memory=config.get('pin_memory', True))
        va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config.get('num_workers', 2), pin_memory=config.get('pin_memory', True))

        opt = optim.AdamW(flow.parameters(), lr=lr_flow, weight_decay=wd_flow)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)

        def loss_fn(y, cond):
            # expects y to be on the unconstrained real line (logit) - already in logit from embeds.npz
            return flow.batch_loss(y, cond)

        best, stale = float('inf'), 0
        train_losses, val_losses = [], []

        # Use appropriate epochs based on inference backend
        inference_epochs = cfm_epochs if inference == 'cfm' else flow_epochs
        
        for ep in range(1, inference_epochs + 1):
            # If this is the FrEIA wrapper it stores the SequenceINN in .flow
            if hasattr(flow, 'flow'):
                try:
                    flow.flow.train()
                except Exception:
                    pass
            elif hasattr(flow, 'train'):
                flow.train()
            running = 0.0
            for c, y in tr:
                c = c.to(device)
                y = y.to(device)
                opt.zero_grad()
                l = loss_fn(y, c)
                l.backward(); opt.step()
                running += l.item()
            train_losses.append(running / len(tr))

            if hasattr(flow, 'flow'):
                try:
                    flow.flow.eval()
                except Exception:
                    pass
            elif hasattr(flow, 'eval'):
                flow.eval()
            running = 0.0
            with torch.no_grad():
                for c, y in va:
                    c = c.to(device)
                    y = y.to(device)
                    running += loss_fn(y, c).item()
            val_losses.append(running / len(va))

            logging.info(f"[FLOW] Ep{ep}: train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")
            sch.step(val_losses[-1])
            if val_losses[-1] < best:
                best = val_losses[-1]; stale = 0
                # save state dict depending on backend
                if inference == 'flow':
                    torch.save(model_state_dict_for_saving(flow, 'flow'), os.path.join(output_dir, 'best_flow_model.pth'))
                else:
                    torch.save(model_state_dict_for_saving(flow, 'cfm'), os.path.join(output_dir, 'best_cfm_model.pth'))
            else:
                stale += 1
                if stale > 10:
                    break

        # save final
        if inference == 'flow':
            torch.save(model_state_dict_for_saving(flow, 'flow'), os.path.join(output_dir, 'final_flow_model.pth'))
        else:
            torch.save(model_state_dict_for_saving(flow, 'cfm'), os.path.join(output_dir, 'final_cfm_model.pth'))
        plt.plot(train_losses, label='train'); plt.plot(val_losses, label='val'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'flow_loss.pdf'))
        np.save(os.path.join(output_dir, 'flow_train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(output_dir, 'flow_val_losses.npy'), np.array(val_losses))

    def finetune_joint(cnn, flow, train_loader, val_loader, output_dir):
        # load best checkpoints if available (look in the stage-specific folders)
        cnn_ckpt = os.path.join(cnn_stage_dir, 'best_cnn_model.pth')
        # prefer backend-appropriate flow checkpoint (search flow_stage_dir first, then finetune_stage_dir)
        if inference == 'cfm':
            candidate_flow_ckpts = [os.path.join(flow_stage_dir, 'best_cfm_model.pth'), os.path.join(finetune_stage_dir, 'best_joint.pth'), os.path.join(flow_stage_dir, 'best_flow_model.pth')]
        else:
            candidate_flow_ckpts = [os.path.join(flow_stage_dir, 'best_flow_model.pth'), os.path.join(finetune_stage_dir, 'best_joint.pth'), os.path.join(flow_stage_dir, 'best_cfm_model.pth')]

        if os.path.exists(cnn_ckpt):
            cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device))

        flow_ckpt = None
        for pth in candidate_flow_ckpts:
            if os.path.exists(pth):
                flow_ckpt = pth
                break

        if flow_ckpt is not None:
            ck = torch.load(flow_ckpt, map_location=device)
            # ck could be a raw state_dict, a dict with 'inference_state_dict', or a joint ck with 'flow' or 'inference_state_dict'
            if isinstance(ck, dict):
                if 'inference_state_dict' in ck:
                    flow.load_state_dict(ck['inference_state_dict'])
                elif 'flow' in ck and isinstance(ck['flow'], dict):
                    flow.load_state_dict(ck['flow'])
                elif 'cnn' in ck and 'flow' in ck:
                    # joint checkpoint saved by this script
                    sub = ck.get('flow') or ck.get('inference_state_dict')
                    if isinstance(sub, dict):
                        flow.load_state_dict(sub)
                    else:
                        try:
                            flow.load_state_dict(ck)
                        except Exception:
                            pass
                else:
                    try:
                        flow.load_state_dict(ck)
                    except Exception:
                        pass

        # warm-up: freeze cnn, train flow only for a few epochs
        for p in cnn.parameters():
            p.requires_grad = False
        cnn.eval()

        opt_flow = optim.AdamW(flow.parameters(), lr=lr_flow, weight_decay=wd_flow)

        for ep in range(1, flow_warmup_epochs + 1):
            if hasattr(flow, 'flow'):
                try:
                    flow.flow.train()
                except Exception:
                    pass
            elif hasattr(flow, 'train'):
                flow.train()
            running = 0.0
            for img, y in train_loader:
                img = img.to(device)
                y = y.to(device)
                opt_flow.zero_grad()
                with torch.no_grad():
                    cond = cnn(img.unsqueeze(1))
                loss = flow.batch_loss(y, cond)
                loss.backward(); opt_flow.step()
                running += loss.item()
            logging.info(f"[WARMUP] Epoch {ep}: flow-train={running/len(train_loader):.6f}")

        # unfreeze last CNN layers 
            for name, p in cnn.named_parameters():
                if 'out' in name or 'linear3' in name:
                    p.requires_grad = True

            opt_cnn = optim.AdamW([p for p in cnn.parameters() if p.requires_grad], lr=finetune_lr, weight_decay=wd_cnn)
            opt_flow = optim.AdamW(flow.parameters(), lr=lr_flow, weight_decay=wd_flow)
            sch = optim.lr_scheduler.ReduceLROnPlateau(opt_flow, mode='min', factor=0.5, patience=5)

        best_val = float('inf'); stale = 0
        train_losses, val_losses = [], []

        for ep in range(1, finetune_epochs + 1):
            cnn.train()
            if hasattr(flow, 'flow'):
                try:
                    flow.flow.train()
                except Exception:
                    pass
            elif hasattr(flow, 'train'):
                flow.train()
            run_tr = 0.0
            for img, y in train_loader:
                img = img.to(device)
                y = y.to(device)
                opt_cnn.zero_grad(); opt_flow.zero_grad()
                cond = cnn(img.unsqueeze(1))
                loss = flow.batch_loss(y, cond)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                opt_cnn.step(); opt_flow.step()
                run_tr += loss.item()
            train_losses.append(run_tr / len(train_loader))

            # validate
            cnn.eval()
            if hasattr(flow, 'flow'):
                try:
                    flow.flow.eval()
                except Exception:
                    pass
            elif hasattr(flow, 'eval'):
                flow.eval()
            run_val = 0.0
            with torch.no_grad():
                for img, y in val_loader:
                    img = img.to(device)
                    y = y.to(device)
                    cond = cnn(img.unsqueeze(1))
                    run_val += (flow.batch_loss(y, cond).item())
            val_loss = run_val / len(val_loader)
            val_losses.append(val_loss)
            logging.info(f"[FINETUNE] Epoch {ep}: tr={train_losses[-1]:.6f}, val={val_loss:.6f}")
            sch.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss; stale = 0
                if inference == 'flow':
                    torch.save({'cnn': cnn.state_dict(), 'flow': model_state_dict_for_saving(flow, 'flow')}, os.path.join(finetune_stage_dir, 'best_joint.pth'))
                else:
                    # ensure we store the raw state dict under 'inference_state_dict'
                    ck = model_state_dict_for_saving(flow, 'cfm')
                    if isinstance(ck, dict) and 'inference_state_dict' in ck:
                        ck_raw = ck['inference_state_dict']
                    else:
                        ck_raw = ck
                    torch.save({'cnn': cnn.state_dict(), 'inference_state_dict': ck_raw}, os.path.join(finetune_stage_dir, 'best_joint.pth'))
            else:
                stale += 1
                if stale > 10:
                    logging.info('Early stopping triggered.'); break

        # save plots and arrays
        np.save(os.path.join(finetune_stage_dir, 'finetune_train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(finetune_stage_dir, 'finetune_val_losses.npy'), np.array(val_losses))
        plt.plot(train_losses, label='train'); plt.plot(val_losses, label='val'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(finetune_stage_dir, 'finetune_loss.pdf'))

    # --- dispatch based on requested stage ---
    if stage == 'cnn':
        logging.info('Starting CNN pretraining stage')
        train_cnn_full(cnn_model, train_loader, val_loader, cnn_stage_dir)
        logging.info('CNN pretraining complete')
        sys.exit(0)

    if stage == 'flow':
        logging.info('Starting FLOW-from-embeds stage')
        embed_path = os.path.join(cnn_stage_dir, 'embeds.npz')
        cnn_ckpt = os.path.join(cnn_stage_dir, 'best_cnn_model.pth')
        # ensure embeddings exist
        if not os.path.exists(embed_path):
            if os.path.exists(cnn_ckpt):
                cnn_model.load_state_dict(torch.load(cnn_ckpt, map_location=device))
                embed_dataset(cnn_model, dataset, cnn_stage_dir)
            else:
                logging.info('No pretrained CNN checkpoint found; training CNN first')
                train_cnn_full(cnn_model, train_loader, val_loader, cnn_stage_dir)
                embed_dataset(cnn_model, dataset, cnn_stage_dir)
        train_flow_from_embeds(flow_model, os.path.join(cnn_stage_dir, 'embeds.npz'), flow_stage_dir)
        logging.info('Flow training from embeddings complete')
        sys.exit(0)

    if stage == 'finetune':
        logging.info('Starting FINETUNE stage (joint cnn+flow)')
        finetune_joint(cnn_model, flow_model, train_loader, val_loader, finetune_stage_dir)
        logging.info('Finetuning complete')
        sys.exit(0)



# ------------------- Power Spectra modes -------------------
else:
    best_val = float('inf')
    patience = 10
    stale = 0

    train_losses, val_losses = [], []

    # use global helpers defined earlier (model_train, model_eval, model_state_dict_for_saving)

    for epoch in range(1, num_epochs+1):
        # — Train —
        model_train(flow_model)
        running_train = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            cond, y = batch
            cond, y = cond.to(device), y.to(device)
            
            # DEBUG: Print shapes on first batch
            if batch_idx == 0 and epoch == 1:
                print(f"\n=== DEBUG SHAPES (epoch 1, batch 0) ===")
                print(f"y.shape (input labels): {y.shape}")
                print(f"cond.shape (conditioning): {cond.shape}")
                print(f"target_dim={target_dim}, cond_dims={cond_dims}")
                print(f"=========================================\n")
            
            loss = flow_loss(flow_model, y, cond, target_dim)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            optimizer.step()
            running_train += loss.item()

        avg_train = running_train / len(train_loader)
        train_losses.append(avg_train)
        # — Validate —
        model_eval(flow_model)
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cond, y = batch
                cond, y = cond.to(device), y.to(device)
                running_val += flow_loss(flow_model, y, cond, target_dim).item()
        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)

        logging.info(f"Epoch {epoch}: Train={avg_train:.6f}, Val={avg_val:.6f}")
        scheduler.step(avg_val)

        # — Early stopping & checkpoint —
        if avg_val < best_val:
            best_val = avg_val
            stale = 0
            # save according to backend
            if inference == 'flow':
                torch.save(model_state_dict_for_saving(flow_model, 'flow'), os.path.join(output_dir,'best_flow_model.pth'))
            else:
                torch.save(model_state_dict_for_saving(flow_model, 'cfm'), os.path.join(output_dir,'best_cfm_model.pth'))
            logging.info(f" New best model at epoch {epoch}")
        else:
            stale += 1
            if stale >= patience:
                logging.info("Early stopping triggered.")
                break

    # ------------------- SAVE FINAL & PLOTS -------------------
    # final save 
    if inference == 'flow':
        torch.save(model_state_dict_for_saving(flow_model, 'flow'), os.path.join(output_dir,'final_flow_model.pth'))
    else:
        torch.save(model_state_dict_for_saving(flow_model, 'cfm'), os.path.join(output_dir,'final_cfm_model.pth'))
    np.save(os.path.join(output_dir,'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(output_dir,'val_losses.npy'),   np.array(val_losses))

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f"EoRFlow ({mode}) — Best Val = {best_val:.4f}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir,'loss_curve.pdf'))