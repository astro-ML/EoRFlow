import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging

def compute_sample_weights(label_means, num_bins=50, eps=1e-6):
    """
    Compute importance weights for a set of scalar target summaries.
    Args:
        label_means (np.ndarray): Array of scalar target values.
        num_bins (int): Number of bins for histogram density estimation.
        eps (float): Small constant to avoid division by zero.
    Returns:
        np.ndarray: Array of importance weights (normalized to have mean 1).
    """
    hist, bin_edges = np.histogram(label_means, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    densities = np.interp(label_means, bin_centers, hist)
    weights = 1.0 / (densities + eps)
    weights = weights / np.mean(weights)
    return weights


class PowerSpectrumDataset(Dataset):
    def __init__(self, data_dirs, xH_constraints=None, skip_fraction=0.0, compute_weights=False, 
                 add_noise=False, augment_noise=False, std_strength=1.0, add_gaussian=False, gaussian_std=0.001, noise_data_dir='/lustre/fswork/projects/rech/ybg/uuv28wh/noise_data',
                 use_cnn=False, k_scale=False, convert_dimensionless=False, logit=True):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally filter samples with xH values outside specified bounds.
        
        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            xH_constraints (dict, optional): Dictionary of specific xH constraints.
                Example: {
                    'xH1_max': 0.8,  # Max value for the first xH (data['label'][0])
                    'xH3_min': 0.2,  # Min value for the third xH (data['label'][2])
                    'global_min': 0.1,  # Min value for any xH
                    'global_max': 0.9,  # Max value for any xH
                }
            skip_fraction (float): Fraction of out-of-bounds samples to drop.
            compute_weights (bool): If True, precompute importance sampling weights.
            add_noise (bool): If True, add realistic noise to power spectra.
            noise_data_dir (str): Directory containing noise mean and std data files.
            use_cnn (bool): If True, return power spectra in 3D format (3,10,10) for CNN processing.
        """
        self.data_dirs = data_dirs
        self.xH_constraints = xH_constraints or {}
        self.skip_fraction = skip_fraction
        self.compute_weights = compute_weights
        self.add_noise = add_noise
        self.augment_noise = augment_noise
        self.std_strength = std_strength
        self.add_gaussian = add_gaussian
        self.gaussian_std = gaussian_std
        self.noise_data_dir = noise_data_dir
        self.use_cnn = use_cnn
        self.k_scale = k_scale
        self.convert_dimensionless = convert_dimensionless
        self.files = []
        self.logit = logit
        
        # Load noise data if needed
        if self.add_noise:
            self._load_noise_data()
        
        # Collect all .npz files from each directory
        for data_dir in self.data_dirs:
            self.files.extend([
                os.path.join(data_dir, f) 
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ])
        logging.info(f"Found {len(self.files)} .npz files across {len(self.data_dirs)} directories.")
        
        # Pre-filter files based on xH constraints if specified
        self.total_files = len(self.files)
        if self.xH_constraints:
            self._prefilter_files()
            excluded_files = self.total_files - len(self.files)
            logging.info(f"xH filtering: {excluded_files} files excluded, {len(self.files)} files kept ({excluded_files/self.total_files:.2%} excluded)")
        
        # If weights are requested, precompute a weight for each sample.
        self.weights = None
        if self.compute_weights:
            label_means = []
            for file_path in self.files:
                data = np.load(file_path)
                # Use only the first three label values (xH values).
                label = data['xH_label'][:3]
                # Compute a scalar summary: the mean.
                label_means.append(np.mean(label))
            label_means = np.array(label_means)
            self.weights = compute_sample_weights(label_means)
    
    def _load_noise_data(self):
        """Load the mean and standard deviation noise data files."""
        try:
            # Load mean power spectra (in order of decreasing frequency / increasing redshift)
            self.mean_ps = np.array([
                np.loadtxt(os.path.join(self.noise_data_dir, 'Pk_PS_averaged_noise_181.0_195.9.txt')),  # z=7.96
                np.loadtxt(os.path.join(self.noise_data_dir, 'Pk_PS_averaged_noise_166.0_180.9.txt')),  # z=7.19
                np.loadtxt(os.path.join(self.noise_data_dir, 'Pk_PS_averaged_noise_151.0_165.9.txt'))   # z=6.54
            ])
            
            # Load standard deviation power spectra (in order of decreasing frequency / increasing redshift)
            self.std_ps = np.array([
                np.loadtxt(os.path.join(self.noise_data_dir, 'noise_std_bin1')),  # z=7.96
                np.loadtxt(os.path.join(self.noise_data_dir, 'noise_std_bin2')),  # z=7.19
                np.loadtxt(os.path.join(self.noise_data_dir, 'noise_std_bin3'))   # z=6.54
            ])
            
            # Verify shapes are correct
            if self.mean_ps.shape != (3, 10, 10) or self.std_ps.shape != (3, 10, 10):
                raise ValueError(f"Unexpected noise data shapes: mean={self.mean_ps.shape}, std={self.std_ps.shape}. Expected (3, 10, 10).")
            
            logging.info("Noise data loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading noise data: {e}")
            self.add_noise = False
            logging.warning("Noise addition disabled due to error")
    
    def sample_noise(self):
        """
        Sample noise from Gaussian distributions based on mean and standard deviation data.
        
        Returns:
        --------
        noise : numpy.ndarray
            Sampled noise with shape (3, 10, 10)
        """
        # Generate random samples from a standard normal distribution
        random_samples = np.random.normal(0, 1, size=(3, 10, 10))
        
        # Scale by standard deviation and shift by mean
        if self.augment_noise:
            #logging.info('Training with augmented noise...')
            noise = self.mean_ps + random_samples * self.std_ps * self.std_strength
        else:
            noise = self.mean_ps
            #logging.info('Training with average noise...')
        return noise
    
    def _prefilter_files(self):
        """
        Pre-filter files based on specific xH constraints.
        Tracks detailed statistics about which constraints filtered out files.
        """
        filtered_files = []
        excluded_count = 0
        exclusion_reasons = {
            'xH1_max': 0,
            'xH2_max': 0,
            'xH3_max': 0, 
            'xH1_min': 0,
            'xH2_min': 0,
            'xH3_min': 0,
            'global_min': 0,
            'global_max': 0,
            'error': 0
        }
        
        print(f"Filtering files with constraints: {self.xH_constraints}")
        for file_path in self.files:
            # Randomly skip based on skip_fraction
            if np.random.rand() >= self.skip_fraction:
                filtered_files.append(file_path)
                continue
                
            try:
                data = np.load(file_path)
                label = data['xH_label'][:3]  # Assuming first 3 elements are xH values
                
                # Check each specific constraint
                excluded = False
                
                # Global min/max constraints (any value outside range)
                if 'global_min' in self.xH_constraints and np.any(label < self.xH_constraints['global_min']):
                    excluded = True
                    exclusion_reasons['global_min'] += 1
                
                if 'global_max' in self.xH_constraints and np.any(label > self.xH_constraints['global_max']):
                    excluded = True
                    exclusion_reasons['global_max'] += 1
                
                # Specific xH value constraints
                for i, xh_name in enumerate(['xH1', 'xH2', 'xH3']):
                    # Check min constraint for this specific xH value
                    min_key = f"{xh_name}_min"
                    if min_key in self.xH_constraints and label[i] < self.xH_constraints[min_key]:
                        excluded = True
                        exclusion_reasons[min_key] += 1
                    
                    # Check max constraint for this specific xH value
                    max_key = f"{xh_name}_max"
                    if max_key in self.xH_constraints and label[i] > self.xH_constraints[max_key]:
                        excluded = True
                        exclusion_reasons[max_key] += 1
                
                if excluded:
                    excluded_count += 1
                else:
                    # If we reach here, the file is within bounds
                    filtered_files.append(file_path)
                    
            except Exception as e:
                logging.warning(f"Error loading {file_path}: {e}")
                exclusion_reasons['error'] += 1
                excluded_count += 1
        
        self.files = filtered_files
        
        # Print detailed statistics
        print(f"Filtering results:")
        print(f"  - Total files: {self.total_files}")
        print(f"  - Files excluded: {excluded_count} ({excluded_count/self.total_files:.2%})")
        
        if excluded_count > 0:
            print(f"  - Exclusion breakdown:")
            for reason, count in exclusion_reasons.items():
                if count > 0:
                    print(f"    - {reason}: {count} ({count/excluded_count:.2%} of excluded)")
        
        print(f"  - Files kept: {len(filtered_files)} ({len(filtered_files)/self.total_files:.2%})")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations and return the power spectrum
        and (optionally) the label. If xH_label isn’t in the file, we return zeros.
        """
        file_path = self.files[idx]
        data = np.load(file_path)
    
        # --- build your ps_tensor exactly as before ---
        k_perp = data['k_perp']
        k_par  = data['k_par']
        ps     = data['power_spectra'].copy()  # (3,10,10)
    
        if self.add_gaussian:
            ps += np.random.normal(0, self.gaussian_std, size=ps.shape)
        if self.add_noise:
            ps += self.sample_noise()
        if self.k_scale:
            w = np.exp(-3*(k_perp**2 + k_par**2))
            ps *= w
        if self.convert_dimensionless:
            ps = (np.sqrt(np.add.outer(k_perp**2, k_par**2))**3/(2*np.pi))*ps
    
        if self.use_cnn:
            # per‐slice normalization
            for i in range(ps.shape[0]):
                sl = ps[i]
                ps[i] = (sl - sl.min())/(sl.max()-sl.min()+1e-6)
            ps_tensor = torch.tensor(ps, dtype=torch.float32)
    
        else:
            # flat + redshift conditioning
            ps = (ps - ps.min())/(ps.max()-ps.min()+1e-6)
            zs = np.array([6.54,7.19,7.96])
            zs = (zs - zs.min())/(zs.max()-zs.min())
            cond = []
            for i in range(3):
                cond.append( np.concatenate([ps[i].ravel(), zs[[i]]]) )
            ps_tensor = torch.tensor(np.concatenate(cond), dtype=torch.float32)
    
        # --- now optional label loading ---
        if 'xH_label' in data.files:
            label = data['xH_label']
            if self.logit:
                eps = 1e-5
                label = eps + (1-2*eps)*label
                label = np.log(label/(1-label))
        else:
            # fallback: zero‐vector of length 3
            label = np.zeros(3, dtype=np.float32)
    
        label_tensor = torch.tensor(label, dtype=torch.float32)
    
        if self.compute_weights:
            weight = torch.tensor(self.weights[idx], dtype=torch.float32)
            return ps_tensor, label_tensor, weight
        else:
            return ps_tensor, label_tensor

###########################################################################################
class SortedPowerSpectrumDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

        # Create a list of indices corresponding to the samples
        self.indices = list(range(len(original_dataset)))

        # Sort the indices based on the file names
        self.indices.sort(key=lambda idx: self.original_dataset.files[idx], reverse=True)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        # Get the sorted index
        sorted_index = self.indices[index]
        # Return the data and label from the original dataset
        return self.original_dataset[sorted_index]
