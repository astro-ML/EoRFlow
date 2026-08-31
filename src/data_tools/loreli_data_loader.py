"""
Data loader for Loreli II dataset, rebinned to match EoRFlow ps1d format.
Handles power spectrum rebinning from 512 linear k-bins to 14 irregular k-bins,
redshift matching, and neutral fraction label extraction from diagnostics files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from pathlib import Path


class LoreliPS1DDataset(Dataset):
    """
    Dataset loader for Loreli II power spectra in EoRFlow-compatible format.
    
    Loreli II structure:
    - powerspectra/simuNNNNN/postprocessing/ps_dtb/powerspectrum*.dat
    - powerspectra/simuNNNNN/redshift_list.dat
    - diagnostics/simuNNNNN/diagnostics.dat
    
    EoRFlow target format:
    - ps1d: (n_redshifts, 14) array
    - xHI_labels: (n_redshifts,) array
    - k-bins: 14 irregular bins [0, 0.054, ..., 2.96] h/Mpc
    - z-range: [6.50, 12.00] (12 slices, skipping z<6.30 not in Loreli)
    """
    
    # EoRFlow target k-bins (irregular, log-ish spacing) in h/Mpc
    # Both Loreli and 21cmFAST data use h/Mpc units for k-bins
    EORFLOW_K_BINS = np.array([
        0.0, 0.05441398, 0.07695299, 0.10841858, 0.15415038, 0.21540695,
        0.3018365, 0.42428553, 0.59825265, 0.8431039, 1.1871009,
        1.6723664, 2.3015926, 2.9585834
    ], dtype=np.float32)
    
    # EoRFlow target redshifts (we'll use 12 of these: z >= 6.50)
    EORFLOW_REDSHIFTS = np.array([
        5.233072, 5.4996195, 6.0014706, 6.4982133, 7.00021, 7.5001707,
        8.000271, 8.499395, 8.997384, 9.501907, 10.002276, 10.500712,
        11.000936, 11.499588, 12.002204
    ], dtype=np.float32)
    
    def __init__(
        self,
        loreli_root,
        mode='ps1d',
        min_redshift_index=3,    # Start at z=6.50 (index 3), skip z<6.30
        max_redshift_index=15,   # Up to z=12.00
        logit=True,
        add_noise=False,
        num_samples=None,
        verbose=False,
    ):
        """
        Args:
            loreli_root: Path to Loreli_II directory containing powerspectra/ and diagnostics/
            mode: Only 'ps1d' supported (for compatibility with EoRFlow)
            min_redshift_index: Start index in EORFLOW_REDSHIFTS (3 = z=6.50)
            max_redshift_index: End index in EORFLOW_REDSHIFTS (15 = z=12.00)
            logit: Apply logit transform to xHI labels
            add_noise: Add Gaussian noise to power spectra
            num_samples: Limit number of simulations (None = all)
            verbose: Print debug information
        """
        assert mode == 'ps1d', "Only 'ps1d' mode supported for Loreli data"
        
        self.loreli_root = Path(loreli_root)
        self.mode = mode
        self.logit = logit
        self.add_noise = add_noise
        self.min_z = min_redshift_index
        self.max_z = max_redshift_index
        self.verbose = verbose
        
        # Target redshifts (subset that overlaps with Loreli)
        self.ps_redshifts = self.EORFLOW_REDSHIFTS[self.min_z:self.max_z]
        self.n_redshifts = len(self.ps_redshifts)
        
        # Store k-bins for reference (both Loreli and 21cmFAST use h/Mpc)
        self.ps_k_bins = self.EORFLOW_K_BINS
        
        # Find all valid simulations
        self.valid_simulations = self._find_valid_simulations()
        
        if num_samples is not None:
            self.valid_simulations = self.valid_simulations[:num_samples]
        
        if len(self.valid_simulations) == 0:
            raise ValueError(f"No valid simulations found in {loreli_root}")
        
        if self.verbose:
            print(f"Loaded {len(self.valid_simulations)} valid Loreli simulations")
            print(f"Using {self.n_redshifts} redshift slices: z=[{self.ps_redshifts.min():.2f}, {self.ps_redshifts.max():.2f}]")
    
    def _find_valid_simulations(self):
        """
        Find all simulations that have:
        1. Power spectra directory with powerspectrum files
        2. Redshift list file
        3. Diagnostics file
        4. All required redshift power spectra files
        """
        valid_sims = []
        
        ps_dir = self.loreli_root / "powerspectra"
        diag_dir = self.loreli_root / "diagnostics"
        
        if not ps_dir.exists():
            raise ValueError(f"Power spectra directory not found: {ps_dir}")
        if not diag_dir.exists():
            raise ValueError(f"Diagnostics directory not found: {diag_dir}")
        
        # Get all simulation directories
        sim_dirs = sorted([d for d in ps_dir.iterdir() if d.is_dir() and d.name.startswith('simu')])
        
        for sim_dir in sim_dirs:
            sim_id = sim_dir.name
            
            # Check for required files
            redshift_file = sim_dir / "redshift_list.dat"
            ps_subdir = sim_dir / "postprocessing" / "ps_dtb"
            diag_file = diag_dir / sim_id / "diagnostics.dat"
            
            if not redshift_file.exists():
                continue
            if not ps_subdir.exists():
                continue
            if not diag_file.exists():
                continue
            
            # Load redshift list to check coverage
            try:
                redshift_data = np.loadtxt(redshift_file)
                
                # Handle single-redshift files (1D array instead of 2D)
                if redshift_data.ndim == 1:
                    if len(redshift_data) >= 2:
                        redshift_data = redshift_data.reshape(1, -1)
                    else:
                        continue  # Skip files with insufficient data
                
                z_loreli = redshift_data[:, 0]
                file_indices = redshift_data[:, 1].astype(int)
                
                # Check if we can map all target redshifts
                can_map_all = True
                for z_target in self.ps_redshifts:
                    # Find closest Loreli redshift
                    closest_idx = np.argmin(np.abs(z_loreli - z_target))
                    z_closest = z_loreli[closest_idx]
                    file_idx = file_indices[closest_idx]
                    
                    # Check if the power spectrum file exists and is not empty
                    ps_file = ps_subdir / f"powerspectrum{file_idx:03d}.dat"
                    if not ps_file.exists() or ps_file.stat().st_size == 0:
                        can_map_all = False
                        break
                    
                    # Check if match is reasonable (within 0.5 in redshift)
                    if abs(z_closest - z_target) > 0.5:
                        can_map_all = False
                        break
                
                if can_map_all:
                    valid_sims.append(sim_id)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Skipping {sim_id}: {e}")
                continue
        
        return valid_sims
    
    def _rebin_ps1d(self, k_loreli, pk_loreli):
        """
        Rebin Loreli power spectrum from 512 linear bins to 14 irregular bins.
        Uses bin-averaging for noise reduction.
        
        Args:
            k_loreli: (512,) array of k-values in h/Mpc
            pk_loreli: (512,) array of P(k) values in mK²
        
        Returns:
            ps_rebinned: (14,) array matching EORFLOW_K_BINS
        """
        ps_rebinned = np.zeros(len(self.EORFLOW_K_BINS), dtype=np.float32)
        
        # k=0 bin: set to zero (Loreli has no DC component)
        ps_rebinned[0] = 0.0
        
        # Rebin remaining bins using averaging
        for i in range(1, len(self.EORFLOW_K_BINS)):
            k_low = self.EORFLOW_K_BINS[i-1]
            k_high = self.EORFLOW_K_BINS[i]
            
            # Find all Loreli k-bins in this range
            mask = (k_loreli >= k_low) & (k_loreli < k_high)
            
            if mask.sum() > 0:
                # Average P(k) values in this bin
                ps_rebinned[i] = np.mean(pk_loreli[mask])
            else:
                # No Loreli bins in this range - use nearest neighbor
                closest_idx = np.argmin(np.abs(k_loreli - (k_low + k_high) / 2))
                ps_rebinned[i] = pk_loreli[closest_idx]
        
        return ps_rebinned
    
    def _load_simulation_data(self, sim_id):
        """
        Load and process all data for one simulation.
        
        Returns:
            ps1d: (n_redshifts, 14) array of rebinned power spectra
            xHI_labels: (n_redshifts,) array of neutral fractions
        """
        sim_dir = self.loreli_root / "powerspectra" / sim_id
        ps_subdir = sim_dir / "postprocessing" / "ps_dtb"
        diag_file = self.loreli_root / "diagnostics" / sim_id / "diagnostics.dat"
        
        # Load redshift mapping
        redshift_data = np.loadtxt(sim_dir / "redshift_list.dat")
        
        # Handle single-redshift files
        if redshift_data.ndim == 1:
            redshift_data = redshift_data.reshape(1, -1)
        
        z_loreli = redshift_data[:, 0]
        file_indices = redshift_data[:, 1].astype(int)
        
        # Load diagnostics (z, x_e, T, ...)
        diagnostics = np.loadtxt(diag_file)
        diag_z = diagnostics[:, 0]
        diag_xe = diagnostics[:, 1]  # ionized fraction
        diag_xHI = 1.0 - diag_xe      # neutral fraction
        
        # Arrays to store results
        ps1d_list = []
        xHI_list = []
        
        # Process each target redshift
        for z_target in self.ps_redshifts:
            # Find closest Loreli power spectrum redshift
            closest_ps_idx = np.argmin(np.abs(z_loreli - z_target))
            z_ps = z_loreli[closest_ps_idx]
            file_idx = file_indices[closest_ps_idx]
            
            # Load power spectrum file
            ps_file = ps_subdir / f"powerspectrum{file_idx:03d}.dat"
            try:
                ps_data = np.loadtxt(ps_file)
                if ps_data.size == 0 or ps_data.ndim != 2 or ps_data.shape[1] < 2:
                    raise ValueError(f"Invalid power spectrum data shape: {ps_data.shape}")
                k_loreli = ps_data[:, 0]
                pk_loreli = ps_data[:, 1]
            except (ValueError, IndexError) as e:
                raise RuntimeError(f"Failed to load {ps_file} for {sim_id}: {e}")
            
            # Rebin to EoRFlow format
            ps_rebinned = self._rebin_ps1d(k_loreli, pk_loreli)
            ps1d_list.append(ps_rebinned)
            
            # Find closest diagnostics redshift for label
            closest_diag_idx = np.argmin(np.abs(diag_z - z_ps))
            xHI_val = diag_xHI[closest_diag_idx]
            xHI_list.append(xHI_val)
        
        ps1d = np.stack(ps1d_list, axis=0)  # (n_redshifts, 14)
        xHI_labels = np.array(xHI_list, dtype=np.float32)  # (n_redshifts,)
        
        return ps1d, xHI_labels
    
    def __len__(self):
        return len(self.valid_simulations)
    
    def __getitem__(self, idx):
        """
        Get one simulation's data in EoRFlow format.
        
        Returns:
            cond: Flattened conditioning array (ps1d + normalized redshifts)
            lbl: xHI labels, optionally logit-transformed
        """
        sim_id = self.valid_simulations[idx]
        
        # Load and rebin data
        ps1d, xHI_labels = self._load_simulation_data(sim_id)
        
        # Add Gaussian noise if requested
        if self.add_noise:
            ps1d = ps1d + np.random.normal(0, 0.05, ps1d.shape).astype(np.float32)
        
        # Normalize power spectrum (per-sample min-max normalization)
        ps1d = (ps1d - np.min(ps1d)) / (np.max(ps1d) - np.min(ps1d) + 1e-6)
        
        # Build conditioning array: flatten ps1d per redshift + append normalized z
        cond = []
        zs = self.ps_redshifts
        for i, zval in enumerate(zs):
            slice_i = ps1d[i].flatten()  # (14,)
            znorm = (zval - zs.min()) / (zs.max() - zs.min() + 1e-6)
            cond.append(np.concatenate([slice_i, [znorm]]))
        cond = np.stack(cond, axis=0)  # (n_redshifts, 15)
        cond = cond.reshape(-1)  # flatten all together
        
        # Apply logit transform to labels if requested
        lbl = xHI_labels.copy()
        if self.logit:
            eps = 1e-5
            x = eps + (1 - 2*eps) * lbl
            lbl = np.log(x / (1 - x))
        
        return (
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.float32),
        )


def test_loreli_loader():
    """Quick test function to verify data loading."""
    import matplotlib.pyplot as plt
    
    loreli_root = "/pfs/10/work/hd_pt254-skatr/Loreli_II"
    
    print("Testing LoreliPS1DDataset...")
    dataset = LoreliPS1DDataset(
        loreli_root,
        min_redshift_index=3,
        max_redshift_index=15,
        verbose=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load first sample
    cond, lbl = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Conditioning shape: {cond.shape}")
    print(f"  Labels shape: {lbl.shape}")
    print(f"  Label values (logit): {lbl.numpy()}")
    
    # Inverse logit to check xHI values
    eps = 1e-5
    lbl_np = lbl.numpy()
    xHI_recovered = (np.exp(lbl_np) - eps) / (1 - 2*eps + np.exp(lbl_np))
    print(f"  Label values (xHI): {xHI_recovered}")
    
    # Load a few more samples to check consistency
    print("\nChecking consistency across samples...")
    for i in range(min(5, len(dataset))):
        cond_i, lbl_i = dataset[i]
        print(f"  Sample {i}: cond shape={cond_i.shape}, lbl shape={lbl_i.shape}")
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_loreli_loader()
