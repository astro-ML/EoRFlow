import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging

class PowerSpectrumDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        min_redshift_index=0,
        max_redshift_index=15,
        max_zeros_allowed=15,
        max_ones_allowed=15,
        filter_reionization_timing=False,  
        use_cnn=False,
        logit=False,
        add_noise=False,
        num_files=None,  
    ):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Filter out files based on specified criteria and optionally limit the
        number of files loaded.
        
        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            max_redshift_index (int): Number of redshift slices to keep (from the beginning).
            max_zeros_allowed (int): Maximum number of 0 values allowed in the xH values.
            max_ones_allowed (int): Maximum number of 1 values allowed in the xH values.
            filter_reionization_timing (bool): Whether to filter out late reionizations
                                               (xH(z=5) > 0.01) and early reionizations 
                                               (xH(z=12) < 0.9).
            use_cnn (bool): Whether to use CNN formatting for the data.
            logit (bool): Whether to apply the logit transformation to the labels.
            add_noise (bool): If true, adds Gaussian noise to the power spectrum.
            num_files (int or None): Maximum number of files to load; default is None (load all files).
        """
        self.data_dirs = data_dirs
        self.min_redshift_index = min_redshift_index
        self.max_redshift_index = max_redshift_index
        self.max_zeros_allowed = max_zeros_allowed
        self.max_ones_allowed = max_ones_allowed
        self.filter_reionization_timing = filter_reionization_timing
        self.use_cnn = use_cnn
        self.logit = logit
        self.add_noise = add_noise
        self.num_files = num_files  
        
        # Collect all .npz files from each directory
        all_files = []
        for data_dir in self.data_dirs:
            all_files.extend([
                os.path.join(data_dir, f) 
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ])

        logging.info(f"Found {len(all_files)} .npz files across {len(self.data_dirs)} directories.")

        # Load redshift values from the first file
        first_file = all_files[0]
        data = np.load(first_file)
        self.redshift_values = data['redshifts'][min_redshift_index:max_redshift_index]
        
        # Filter files based on the number of 0s and 1s in xH values
        filtered_files = self._filter_by_extreme_values(all_files)
        
        # Filter files based on reionization timing if enabled
        if self.filter_reionization_timing:
            filtered_files = self._filter_by_reionization_timing(filtered_files)
        
        # Limit the number of files if num_files is provided
        if self.num_files is not None:
            filtered_files = filtered_files[-self.num_files:]
            logging.info(f"Limiting the dataset to the first {self.num_files} files.")

        self.files = filtered_files
        
        logging.info(f"Keeping {len(self.files)} files after all filtering steps.")

    def _filter_by_extreme_values(self, file_list):
        """
        Filter files based on the maximum number of 0s and 1s allowed in xH values.
        
        Args:
            file_list (list): List of file paths to filter.
            
        Returns:
            list: Filtered list of file paths.
        """
        filtered_files = []
        excluded_zeros = 0
        excluded_ones = 0
        excluded_both = 0
        
        for file_path in file_list:
            data = np.load(file_path)
            xH_values = data['label'][self.min_redshift_index:self.max_redshift_index]
            
            # Count zeros and ones
            zeros_count = np.sum(xH_values == 0)
            ones_count = np.sum(xH_values == 1)
            
            # Check if the file meets the criteria
            zeros_ok = zeros_count <= self.max_zeros_allowed
            ones_ok = ones_count <= self.max_ones_allowed
            
            if zeros_ok and ones_ok:
                filtered_files.append(file_path)
            else:
                if not zeros_ok and not ones_ok:
                    excluded_both += 1
                elif not zeros_ok:
                    excluded_zeros += 1
                else:
                    excluded_ones += 1
        
        logging.info(f"Excluded {excluded_zeros} files due to too many zeros.")
        logging.info(f"Excluded {excluded_ones} files due to too many ones.")
        logging.info(f"Excluded {excluded_both} files due to both too many zeros and ones.")
        
        return filtered_files

    def _filter_by_reionization_timing(self, file_list):
        """
        Filter files based on reionization timing criteria:
        - Exclude late reionizations: xH(z=5) > 0.01
        - Exclude early reionizations: xH(z=12) < 0.9
        
        Args:
            file_list (list): List of file paths to filter.
            
        Returns:
            list: Filtered list of file paths.
        """
        filtered_files = []
        excluded_late = 0
        excluded_early = 0
        excluded_both_timing = 0
        
        for file_path in file_list:
            data = np.load(file_path)
            
            # Get redshift values and xH values
            redshifts = data['redshifts'][self.min_redshift_index:self.max_redshift_index]
            xH_values = data['label'][self.min_redshift_index:self.max_redshift_index]
            
            # Find indices for z=5 and z=12 (or the closest values)
            z5_idx = np.argmin(np.abs(redshifts - 5))
            z12_idx = np.argmin(np.abs(redshifts - 12))
            
            # Get the corresponding xH values
            xH_at_z5 = xH_values[z5_idx]
            xH_at_z12 = xH_values[z12_idx]
            
            # Check timing criteria
            late_reionization = xH_at_z5 = 0.0
            early_reionization = xH_at_z12 < 0.9
            
            # Apply the filter
            if not late_reionization and not early_reionization:
                filtered_files.append(file_path)
            else:
                if late_reionization and early_reionization:
                    excluded_both_timing += 1
                elif late_reionization:
                    excluded_late += 1
                else:
                    excluded_early += 1
        
        logging.info(f"Excluded {excluded_late} files due to late reionization (xH(z=5) > 0.01).")
        logging.info(f"Excluded {excluded_early} files due to early reionization (xH(z=12) < 0.9).")
        logging.info(f"Excluded {excluded_both_timing} files due to both late and early reionization.")
        
        return filtered_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply normalization, and return
        the power spectrum, label, and redshift values.
        
        Returns:
            ps_tensor (torch.Tensor): 3D power spectrum (shape (max_redshift_index, 10, 10)).
            label_tensor (torch.Tensor): xH values (shape (max_redshift_index,)).
            redshifts_tensor (torch.Tensor): Redshift values (shape (max_redshift_index,)).
        """
        file_path = self.files[idx]
        data = np.load(file_path)
       
        # Get power spectrum data and apply min-max normalization
        ps = data['image'][self.min_redshift_index:self.max_redshift_index, :, :]
        if self.add_noise:
            noise_std = 0.05
            # Generate noise with the same shape as the dataset
            noise = np.random.normal(loc=0, scale=noise_std, size=ps.shape)
            ps = ps + noise

        ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)
        
        # Get label (xH values)
        label = data['label'][self.min_redshift_index:self.max_redshift_index]
        
        if self.logit:
            # Squeeze to avoid boundaries
            epsilon = 1e-5
            label = epsilon + (1 - 2 * epsilon) * label
            # Apply logit transformation
            label = np.log(label / (1 - label))
        
        # Get and normalize redshifts
        redshifts = data['redshifts'][self.min_redshift_index:self.max_redshift_index]
        redshifts = (redshifts - np.min(redshifts)) / (np.max(redshifts) - np.min(redshifts) + 1e-6)
        
        if not self.use_cnn:
            # Append redshift to the power spectrum
            cond = []
            for i in range(len(redshifts)):
                ps_slice = ps[i].flatten()
                z_val = redshifts[np.newaxis, i]
                cond.append(np.concatenate([ps_slice, z_val]))
            condition = np.concatenate(cond)
            # Convert to PyTorch tensors
            condition_tensor = torch.tensor(condition, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return condition_tensor, label_tensor
        else:
            # Convert to PyTorch tensors
            ps_tensor = torch.tensor(ps, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            redshifts_tensor = torch.tensor(redshifts, dtype=torch.float32)
            return ps_tensor, label_tensor, redshifts_tensor

    def get_histogram_data(self):
        """
        Calculate histogram data for xH values across all files.
        
        Returns:
            dict: Dictionary containing histogram data and statistics.
        """
        all_xh_values = []
        
        for file_path in self.files:
            data = np.load(file_path)
            xh = data['label'][self.min_redshift_index:self.max_redshift_index]
            all_xh_values.extend(xh)
        
        all_xh_values = np.array(all_xh_values)
        
        # Calculate basic statistics
        stats = {
            'min': float(np.min(all_xh_values)),
            'max': float(np.max(all_xh_values)),
            'mean': float(np.mean(all_xh_values)),
            'median': float(np.median(all_xh_values)),
            'zeros_percentage': float(100 * np.sum(all_xh_values == 0) / len(all_xh_values)),
            'ones_percentage': float(100 * np.sum(all_xh_values == 1) / len(all_xh_values)),
            'total_values': len(all_xh_values),
            'total_files': len(self.files)
        }
        
        # Create histogram data
        hist, bin_edges = np.histogram(all_xh_values, bins=20, range=(0, 1))
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'statistics': stats
        }
        
    def get_reionization_timing_stats(self):
        """
        Calculate statistics related to reionization timing.
        
        Returns:
            dict: Dictionary containing reionization timing statistics.
        """
        z5_xh_values = []
        z12_xh_values = []
        
        for file_path in self.files:
            data = np.load(file_path)
            redshifts = data['redshifts'][self.min_redshift_index:self.max_redshift_index]
            xH_values = data['label'][self.min_redshift_index:self.max_redshift_index]
            
            # Find indices for z=5 and z=12 (or the closest values)
            z5_idx = np.argmin(np.abs(redshifts - 5))
            z12_idx = np.argmin(np.abs(redshifts - 12))
            
            # Get the corresponding xH values
            z5_xh_values.append(xH_values[z5_idx])
            z12_xh_values.append(xH_values[z12_idx])
        
        z5_xh_values = np.array(z5_xh_values)
        z12_xh_values = np.array(z12_xh_values)
        
        return {
            'z5_mean_xh': float(np.mean(z5_xh_values)),
            'z5_median_xh': float(np.median(z5_xh_values)),
            'z5_min_xh': float(np.min(z5_xh_values)),
            'z5_max_xh': float(np.max(z5_xh_values)),
            'z12_mean_xh': float(np.mean(z12_xh_values)),
            'z12_median_xh': float(np.median(z12_xh_values)),
            'z12_min_xh': float(np.min(z12_xh_values)),
            'z12_max_xh': float(np.max(z12_xh_values)),
            'late_reionization_percentage': float(100 * np.sum(z5_xh_values > 0.01) / len(z5_xh_values)),
            'early_reionization_percentage': float(100 * np.sum(z12_xh_values < 0.9) / len(z12_xh_values))
        }
