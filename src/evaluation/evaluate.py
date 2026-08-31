#!/usr/bin/env python
"""
Unified evaluation script for EoRFlow models.
Works with both 21cmFAST and Loreli datasets.
"""

import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Add paths
sys.path.append('/pfs/10/work/hd_pt254-eorflow/EoRFlow-dev/src/models')
sys.path.append('/pfs/10/work/hd_pt254-eorflow/EoRFlow-dev/src/data_tools')
sys.path.append('/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/src/data_tools')

from flow import ConditionalInvertibleBlock
from data_loader import EoRH5Dataset
from loreli_data_loader import LoreliPS1DDataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataset(config):
    """Create dataset based on config."""
    data = config['data']
    
    if data == 'loreli':
        # Loreli dataset
        dataset = LoreliPS1DDataset(
            loreli_root=config.get('loreli_root', '/pfs/10/work/hd_pt254-skatr/Loreli_II'),
            min_redshift_index=config['min_redshift_index'],
            max_redshift_index=config['max_redshift_index'],
            logit=True,
            add_noise=False,
            verbose=False,
        )
        print(f"Loaded Loreli dataset: {len(dataset)} samples")
    else:
        # 21cmFAST or other EoRFlow datasets
        database_path = config.get('database_path', '/pfs/10/project/bw24d007/EoRFlow_storage/')
        data_dirs = [os.path.join(database_path, data, f'batch_{i}') for i in range(0, 4)]
        
        dataset = EoRH5Dataset(
            data_dirs=data_dirs,
            mode=config['mode'],
            min_redshift_index=config['min_redshift_index'],
            max_redshift_index=config['max_redshift_index'],
            logit=True,
            add_noise=config['add_noise'],
            num_files=None
        )
        print(f"Loaded {data} dataset: {len(dataset)} samples")
    
    return dataset


def evaluate(config):
    """Run evaluation on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = create_dataset(config)
    
    # Load test indices
    test_indices_path = config.get('test_indices_path')
    if test_indices_path and os.path.exists(test_indices_path):
        test_indices = np.load(test_indices_path)
        print(f"Loaded {len(test_indices)} test indices from {test_indices_path}")
    else:
        # Use last 10% as test set
        test_n = int(0.1 * len(dataset))
        test_indices = list(range(len(dataset) - test_n, len(dataset)))
        print(f"Using last {len(test_indices)} samples as test set")
    
    # Setup model
    n_dim = config['max_redshift_index'] - config['min_redshift_index']
    cond_dims = n_dim * 15  # redshifts * (14 k-bins + 1 z)
    
    model_params = {
        'flow': {
            'n_dim': n_dim,
            'n_blocks': config.get('n_blocks', 10),
            'n_nodes': config.get('n_nodes', 512),
            'cond_dims': cond_dims,
            'subnet_depth': config.get('subnet_depth', 3),
            'act': config.get('act', 'relu'),
            'load': False,
            'model_location': None,
        }
    }
    
    print(f"\nModel architecture:")
    print(f"  n_dim: {n_dim}")
    print(f"  n_blocks: {model_params['flow']['n_blocks']}")
    print(f"  n_nodes: {model_params['flow']['n_nodes']}")
    
    # Load model
    model_path = config['model_path']
    print(f"\nLoading model from: {model_path}")
    model = ConditionalInvertibleBlock(model_params)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.flow.eval()
    
    # Evaluate on test set
    num_samples = config.get('num_samples_per_sim', 1000)
    batch_size = config.get('batch_size', 16)
    
    print(f"\nGenerating {num_samples} samples per simulation...")
    
    all_samples = []
    all_labels = []
    all_conds = []
    
    with torch.no_grad():
        for idx in test_indices:
            cond, y_logit = dataset[idx]
            
            # Move to device
            cond = cond.to(device)
            
            # Generate samples (in logit space)
            y_pred_logit = model.sample(num_samples, cond)  # (num_samples, n_dim)
            
            # Convert from logit to xHI space
            eps = 1e-5
            y_pred = (torch.exp(y_pred_logit) - eps) / (1 - 2*eps + torch.exp(y_pred_logit))
            all_samples.append(y_pred.cpu().numpy())  # (num_samples, n_dim)
            
            # Convert labels from logit to xHI space
            y_true = (torch.exp(y_logit) - eps) / (1 - 2*eps + torch.exp(y_logit))
            all_labels.append(y_true.cpu().numpy())  # (n_dim,)
            all_conds.append(cond.cpu().numpy())  # (cond_dims,)
            
            if (len(all_samples)) % 100 == 0:
                print(f"Processed {len(all_samples)} simulations")
    
    # Stack arrays
    # all_samples: list of (num_samples, n_dim) -> stack to (n_test, num_samples, n_dim)
    # then transpose to (num_samples, n_test, n_dim) for plotting.py
    preds = np.stack(all_samples, axis=0).transpose(1, 0, 2)  # (num_samples, n_test, n_dim)
    labels = np.stack(all_labels, axis=0)    # (n_test, n_dim)
    conds = np.stack(all_conds, axis=0)      # (n_test, n_features)
    
    print(f"\nFinal shapes:")
    print(f"  Predictions: {preds.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Conditioning: {conds.shape}")
    
    # Save results
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Include dataset name in filename
    data_name = config['data']
    output_file = os.path.join(output_dir, f'samples_{data_name}.npz')
    np.savez(
        output_file,
        preds=preds,
        labels=labels,
        conds=conds
    )
    print(f"\nSaved samples to: {output_file}")
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate EoRFlow model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to evaluation config file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    samples_file = evaluate(config)
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print(f"Results: {samples_file}")
    print("="*50)
