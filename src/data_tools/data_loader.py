import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging


def compute_slice_weights(label_summaries, num_bins=50, eps=1e-6):
    """
    Compute importance weights for individual xH values for each redshift slice.
    
    Args:
        label_summaries (np.ndarray): Array of shape (N, num_slices) containing xH values.
        num_bins (int): Number of bins for histogram density estimation.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        np.ndarray: Array of shape (N, num_slices) with weights normalized so that the mean for each slice equals 1.
    """
    N, num_slices = label_summaries.shape
    weights = np.zeros_like(label_summaries)
    
    # Compute weights for each redshift slice independently.
    for i in range(num_slices):
        hist, bin_edges = np.histogram(label_summaries[:, i], bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        densities = np.interp(label_summaries[:, i], bin_centers, hist)
        weights[:, i] = 1.0 / (densities + eps)
        weights[:, i] /= np.mean(weights[:, i])
    return weights

class PowerSpectrumDataset_global(Dataset):
    def __init__(
        self,
        data_dirs,
        exclude_unfinished_reionization=False,
        xH_threshold=0.01,
        z_target=5.0, 
        exclude_early_reionization=False,
        z_min=7.0,
        compute_weights=False
    ):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally exclude simulations where reionization has not finished by z_target
        and/or where reionization ends before z_min.
        Optionally compute importance weights for each sample based on a scalar summary of the label.
        
        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            exclude_unfinished_reionization (bool): Exclude simulations where xH > xH_threshold at z = z_target.
            xH_threshold (float): Threshold for xH to consider reionization unfinished.
            z_target (float): Redshift at which to check xH.
            exclude_early_reionization (bool): Exclude simulations where reionization ends before z_min.
            z_min (float): Minimum allowed reionization end redshift.
            compute_weights (bool): If True, compute importance weights based on the target distribution.
        """
        self.data_dirs = data_dirs
        self.compute_weights = compute_weights
        self.files = []

        # Collect all .npz files from each directory.
        for data_dir in self.data_dirs:
            self.files.extend([
                os.path.join(data_dir, f) 
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ])

        logging.info(f"Found {len(self.files)} .npz files across {len(self.data_dirs)} directories.")

        # Load redshift values from the first file.
        first_file = self.files[0]
        data = np.load(first_file)
        self.redshift_values = data['redshifts']  # Expecting shape (30,)
        if self.redshift_values[0] > self.redshift_values[-1]:
            self.redshift_values = self.redshift_values[::-1]

        # Optionally filter based on reionization criteria.
        if exclude_unfinished_reionization:
            self._filter_unfinished_reionization(xH_threshold, z_target)
        if exclude_early_reionization:
            self._filter_early_reionization(z_min)

        if self.compute_weights:
            label_summaries = []
            for file_path in self.files:
                data = np.load(file_path)
                label = data['label'][:15]  # 15 redshift slices
                label_summaries.append(label)
            label_summaries = np.array(label_summaries)  # Shape: (num_simulations, 15)
            self.weights = compute_slice_weights(label_summaries)

    def _filter_unfinished_reionization(self, xH_threshold, z_target):
        filtered_files = []
        num_excluded = 0

        # Find index in redshift_values closest to z_target.
        idx_z = np.argmin(np.abs(self.redshift_values - z_target))
        z_actual = self.redshift_values[idx_z]

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label'][:30]  # xH values (30,)
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]
            xH_at_z = xH[idx_z]
            if xH_at_z <= xH_threshold:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files
        logging.info(f"Excluded {num_excluded} simulations where xH > {xH_threshold} at z = {z_actual}.")
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    def _filter_early_reionization(self, z_min):
        filtered_files = []
        num_excluded = 0

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label'][:30]
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]
            z_end = self.compute_reionization_end_redshift(self.redshift_values, xH)
            if z_end <= z_min:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files
        logging.info(f"Excluded {num_excluded} simulations where EoR ends before z = {z_min}.")
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    @staticmethod
    def compute_reionization_end_redshift(redshift_values, xH_values, threshold=0.1):
        idx = np.where(xH_values <= threshold)[0]
        if idx.size > 0:
            z_end = redshift_values[idx[-1]]
        else:
            z_end = redshift_values[-1]
        return z_end

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations, and return
        the power spectrum, label, and redshift values.
        
        Returns:
            ps_tensor (torch.Tensor): 3D power spectrum (e.g., shape (30, 10, 10)).
            label_tensor (torch.Tensor): xH values (shape (30,)).
            redshifts_tensor (torch.Tensor): Redshift values (shape (30,)).
            weight_tensor (torch.Tensor): Importance weight (scalar) if computed.
        """
        file_path = self.files[idx]
        data = np.load(file_path)
        
        ps = data['image'][:15,:,:]  # Expected shape (30, 10, 10)
  
        # Normalize the 3D power spectrum (e.g., min max).

        ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)

        #ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)
        #ps = ps / (np.max(np.abs(ps)) + 1e-6)
        
        label = data['label'][:15]  # xH values (30,)
    
        redshifts = data['redshifts'][:15]

        redshifts = (redshifts - np.min(redshifts)) / (np.max(redshifts) - np.min(redshifts) + 1e-6)  # Normalize redshifts.
        
        ps_tensor = torch.tensor(ps, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        redshifts_tensor = torch.tensor(redshifts, dtype=torch.float32)
        
        if self.compute_weights:
            weight_tensor = torch.tensor(self.weights[idx], dtype=torch.float32)
            return ps_tensor, label_tensor, redshifts_tensor, weight_tensor
        else:
            return ps_tensor, label_tensor, redshifts_tensor


class PowerSpectrumDataset_global_sample(Dataset):
    def __init__(
        self,
        data_dirs,
        exclude_unfinished_reionization=False,
        xH_threshold=0.01,
        z_target=5.0, 
        exclude_early_reionization=False,
        z_min=7.0,
        compute_weights=False,
        undersample_xH=True,  # New flag to enable undersampling
        target_samples_per_bin=200,  # Number of samples per bin after undersampling
        num_bins=20,  # Number of bins for stratification
    ):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally exclude simulations where reionization has not finished by z_target
        and/or where reionization ends before z_min.
        Optionally compute importance weights for each sample based on a scalar summary of the label.
        Optionally apply stratified undersampling to reduce bias from xH = 0 and xH = 1 cases.

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            exclude_unfinished_reionization (bool): Exclude simulations where xH > xH_threshold at z = z_target.
            xH_threshold (float): Threshold for xH to consider reionization unfinished.
            z_target (float): Redshift at which to check xH.
            exclude_early_reionization (bool): Exclude simulations where reionization ends before z_min.
            z_min (float): Minimum allowed reionization end redshift.
            compute_weights (bool): If True, compute importance weights based on the target distribution.
            undersample_xH (bool): If True, apply stratified undersampling to balance xH distribution.
            target_samples_per_bin (int): Maximum number of samples per bin in undersampling.
            num_bins (int): Number of bins for stratification of xH values.
        """
        self.data_dirs = data_dirs
        self.compute_weights = compute_weights
        self.undersample_xH = undersample_xH
        self.target_samples_per_bin = target_samples_per_bin
        self.num_bins = num_bins
        self.files = []

        # Collect all .npz files from each directory.
        for data_dir in self.data_dirs:
            self.files.extend([
                os.path.join(data_dir, f) 
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ])

        logging.info(f"Found {len(self.files)} .npz files across {len(self.data_dirs)} directories.")

        # Load redshift values from the first file.
        first_file = self.files[0]
        data = np.load(first_file)
        self.redshift_values = data['redshifts']  # Expecting shape (30,)
        if self.redshift_values[0] > self.redshift_values[-1]:
            self.redshift_values = self.redshift_values[::-1]

        # Optionally filter based on reionization criteria.
        if exclude_unfinished_reionization:
            self._filter_unfinished_reionization(xH_threshold, z_target)
        if exclude_early_reionization:
            self._filter_early_reionization(z_min)

        # Apply stratified undersampling
        if self.undersample_xH:
            self._apply_undersampling()

        # If reweighting is enabled, precompute weights.
        self.weights = None
        if self.compute_weights:
            label_summaries = []
            for file_path in self.files:
                data = np.load(file_path)
                label = data['label'][:15]  # Expect full 30 xH values.
                label_summaries.append(np.mean(label))
            label_summaries = np.array(label_summaries)
            self.weights = compute_sample_weights(label_summaries)

    def _filter_unfinished_reionization(self, xH_threshold, z_target):
        filtered_files = []
        num_excluded = 0

        # Find index in redshift_values closest to z_target.
        idx_z = np.argmin(np.abs(self.redshift_values - z_target))
        z_actual = self.redshift_values[idx_z]

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label'][:30]  # xH values (30,)
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]
            xH_at_z = xH[idx_z]
            if xH_at_z <= xH_threshold:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files
        logging.info(f"Excluded {num_excluded} simulations where xH > {xH_threshold} at z = {z_actual}.")
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    def _filter_early_reionization(self, z_min):
        filtered_files = []
        num_excluded = 0

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label'][:30]
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]
            z_end = self.compute_reionization_end_redshift(self.redshift_values, xH)
            if z_end <= z_min:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files
        logging.info(f"Excluded {num_excluded} simulations where EoR ends before z = {z_min}.")
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    @staticmethod
    def compute_reionization_end_redshift(redshift_values, xH_values, threshold=0.1):
        idx = np.where(xH_values <= threshold)[0]
        if idx.size > 0:
            z_end = redshift_values[idx[-1]]
        else:
            z_end = redshift_values[-1]
        return z_end

    def _apply_undersampling(self):
        """
        Stratified undersampling: Reduce overrepresented xH = 0 and xH = 1 cases while maintaining a balanced dataset.
        """
        label_means = []
        filtered_files = []

        # Compute mean xH values for each sample
        for file_path in self.files:
            data = np.load(file_path)
            xH_values = data['label'][:15]  # Consider only first 15 redshifts for undersampling
            mean_xH = np.mean(xH_values)
            label_means.append(mean_xH)

        label_means = np.array(label_means)

        # Define bins for stratification
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(label_means, bins=bin_edges, right=True)

        undersampled_files = []

        # Apply undersampling per bin
        for i in range(1, self.num_bins + 1):
            bin_mask = bin_indices == i
            bin_files = np.array(self.files)[bin_mask]
            
            if len(bin_files) > self.target_samples_per_bin:
                # Randomly sample from the bin to reduce overrepresentation
                selected_files = np.random.choice(bin_files, self.target_samples_per_bin, replace=False)
            else:
                selected_files = bin_files
            
            undersampled_files.extend(selected_files)

        self.files = undersampled_files
        logging.info(f"Applied stratified undersampling. Remaining samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations, and return
        the power spectrum, label, and redshift values.
        
        Returns:
            ps_tensor (torch.Tensor): 3D power spectrum (e.g., shape (30, 10, 10)).
            label_tensor (torch.Tensor): xH values (shape (30,)).
            redshifts_tensor (torch.Tensor): Redshift values (shape (30,)).
            weight_tensor (torch.Tensor): Importance weight (scalar) if computed.
        """
        file_path = self.files[idx]
        data = np.load(file_path)
        
        ps = data['image'][:15,:,:]  # Expected shape (30, 10, 10)
        ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)  # Normalize PS

        label = data['label'][:15]  # xH values (30,)
        redshifts = data['redshifts'][:15]
        redshifts = (redshifts - np.min(redshifts)) / (np.max(redshifts) - np.min(redshifts) + 1e-6)  # Normalize redshifts
        
        ps_tensor = torch.tensor(ps, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        redshifts_tensor = torch.tensor(redshifts, dtype=torch.float32)
        
        if self.compute_weights:
            weight_tensor = torch.tensor(self.weights[idx], dtype=torch.float32)
            return ps_tensor, label_tensor, redshifts_tensor, weight_tensor
        else:
            return ps_tensor, label_tensor, redshifts_tensor



class PowerSpectrumDataset_global_15param(Dataset):
    def __init__(
        self,
        data_dirs,
        exclude_unfinished_reionization=False,
        xH_threshold=0.01,
        z_target=5.5, # 5
        exclude_early_reionization=False,
        z_min=10.0
    ):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally exclude simulations where reionization has not finished by z_target
        (i.e., xH > xH_threshold at z = z_target), and/or exclude simulations where
        reionization ends before z_min (i.e., EoR ends before redshift z_min).

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            exclude_unfinished_reionization (bool): If True, exclude simulations where xH > xH_threshold at z = z_target.
            xH_threshold (float): Threshold for xH to consider reionization unfinished.
            z_target (float): The redshift at which to check xH values.
            exclude_early_reionization (bool): If True, exclude simulations where EoR ends before z_min.
            z_min (float): Minimum allowed reionization end redshift. Simulations with EoR ending before this redshift will be excluded.
        """
        self.data_dirs = data_dirs
        self.files = []

        # Collect all .npz files from each directory
        for data_dir in self.data_dirs:
            self.files.extend(
                [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
            )

        # Debug: Print the total number of files found
        logging.info(
            f"Found {len(self.files)} .npz files in total across {len(self.data_dirs)} directories."
        )

        # Load redshift values from the first file
        first_file = self.files[0]
        data = np.load(first_file)
        self.redshift_values = data['redshifts']  # (30,)

        # Ensure redshift_values are in ascending order
        if self.redshift_values[0] > self.redshift_values[-1]:
            self.redshift_values = self.redshift_values[::-1]

        # Optionally exclude simulations where reionization has not finished by z_target
        if exclude_unfinished_reionization:
            self._filter_unfinished_reionization(xH_threshold, z_target)

        # Optionally exclude simulations where EoR ends before z_min
        if exclude_early_reionization:
            self._filter_early_reionization(z_min)

    def _filter_unfinished_reionization(self, xH_threshold, z_target):
        """
        Filters out simulations where xH > xH_threshold at z = z_target.

        Args:
            xH_threshold (float): Threshold for xH to consider reionization unfinished.
            z_target (float): The redshift at which to check xH values.
        """
        filtered_files = []
        num_excluded = 0

        # Find the index in redshift_values closest to z_target
        idx_z = np.argmin(np.abs(self.redshift_values - z_target))
        z_actual = self.redshift_values[idx_z]

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label']  # xH values (30,)

            # Ensure xH is ordered according to ascending redshift
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]

            xH_at_z = xH[idx_z]

            if xH_at_z <= xH_threshold:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files

        logging.info(
            f"Excluded {num_excluded} simulations where xH > {xH_threshold} at z = {z_actual}."
        )
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    def _filter_early_reionization(self, z_min):
        """
        Filters out simulations where the reionization ends before z_min.

        Args:
            z_min (float): Minimum allowed reionization end redshift. Simulations with EoR ending before this redshift will be excluded.
        """
        filtered_files = []
        num_excluded = 0

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label']  # xH values (30,)

            # Ensure xH is ordered according to ascending redshift
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]

            # Compute reionization end redshift
            z_end = self.compute_reionization_end_redshift(self.redshift_values, xH)

            if z_end <= z_min:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files

        logging.info(
            f"Excluded {num_excluded} simulations where EoR ends before z = {z_min}."
        )
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    @staticmethod
    def compute_reionization_end_redshift(redshift_values, xH_values, threshold=0.1):
        """
        Compute the reionization end redshift (z_end) where xH drops below a threshold.

        Args:
            redshift_values (array): Array of redshift values (ascending order).
            xH_values (array): Corresponding xH values at each redshift.
            threshold (float): xH threshold to consider reionization as ended.

        Returns:
            z_end (float): Reionization end redshift.
        """
        idx = np.where(xH_values <= threshold)[0]
        if idx.size > 0:
            z_end = redshift_values[idx[-1]]  # Highest redshift where xH <= threshold
        else:
            z_end = redshift_values[-1]  # Reionization not ended within the redshift range
        return z_end

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations, and return
        the power spectrum and the corresponding label for CNN.

        Args:
            idx (int): Index of the file to load.

        Returns:
            ps_tensor (torch.Tensor): 3D power spectrum array (30, 10, 10).
            label_tensor (torch.Tensor): Target label (xH values).
        """
        # Load the npz file at the given index
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract the power spectrum and label
        ps = data['image']  # Shape (30, 10, 10)

        # exp: take only one redshift slice
        
        ps = ps[:15,:,:]
    
       
        # Apply normalization to the 3D array
        ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)  # Z-score normalization
        ps = ps / (np.max(np.abs(ps)) + 1e-6)  # Scale to [-1, 1]

        label = data['label'][:15] # xH values (30,)
  
        # Convert to PyTorch tensors
        ps_tensor = torch.tensor(ps, dtype=torch.float32)  # Shape: (30, 10, 10)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return ps_tensor, label_tensor



class PowerSpectrumDataset_SKA(Dataset):
    def __init__(
        self,
        data_dirs,
        exclude_unfinished_reionization=False,
        xH_threshold=0.01,
        z_target=5.0,
        exclude_early_reionization=False,
        z_min=10.0,
        compute_weights=False
    ):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally exclude simulations where reionization has not finished by z_target
        and/or where reionization ends before z_min. Optionally compute importance weights.

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            exclude_unfinished_reionization (bool): If True, exclude simulations where xH > xH_threshold at z = z_target.
            xH_threshold (float): Threshold for xH to consider reionization unfinished.
            z_target (float): The redshift at which to check if reionization is finished.
            exclude_early_reionization (bool): If True, exclude simulations where EoR ends before z_min.
            z_min (float): Minimum allowed reionization end redshift.
            compute_weights (bool): If True, compute importance sampling weights.
        """
        self.data_dirs = data_dirs
        self.compute_weights = compute_weights
        self.files = []

        # Collect all .npz files from each directory.
        for data_dir in self.data_dirs:
            self.files.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])

        logging.info(f"Found {len(self.files)} .npz files in total across {len(self.data_dirs)} directories.")

        if len(self.files) == 0:
            raise ValueError("No .npz files found in the given directories.")

        # For SKA, we set redshift values manually.
        self.redshift_values = np.array([6.54, 7.19, 7.96])
        if self.redshift_values[0] > self.redshift_values[-1]:
            self.redshift_values = self.redshift_values[::-1]

        # Apply filtering if enabled.
        if exclude_unfinished_reionization:
            self._filter_unfinished_reionization(xH_threshold, z_target)
        if exclude_early_reionization:
            self._filter_early_reionization(z_min)

        # If compute_weights is enabled, precompute weights.
        self.weights = None
        if self.compute_weights:
            label_summaries = []
            for file_path in self.files:
                data = np.load(file_path)
                label = data['label'][:3]  # Using the first three xH values.
                # Compute a scalar summary (mean) for reweighting.
                label_summaries.append(np.mean(label))
            label_summaries = np.array(label_summaries)
            self.weights = compute_sample_weights(label_summaries)

    def _filter_unfinished_reionization(self, xH_threshold, z_target):
        filtered_files = []
        num_excluded = 0

        # Find the index in redshift_values closest to z_target.
        idx_z = np.argmin(np.abs(self.redshift_values - z_target))
        z_actual = self.redshift_values[idx_z]

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label']  # Expect shape (3,) for SKA.
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]
            xH_at_z = xH[idx_z]
            if xH_at_z <= xH_threshold:
                filtered_files.append(file_path)
            else:
                num_excluded += 1

        self.files = filtered_files
        logging.info(f"Excluded {num_excluded} simulations where xH > {xH_threshold} at z = {z_actual}.")
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    def _filter_early_reionization(self, z_min):
        filtered_files = []
        num_excluded = 0
        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label']  # Expect shape (3,)
            if self.redshift_values[0] > self.redshift_values[-1]:
                xH = xH[::-1]
            z_end = self.compute_reionization_end_redshift(self.redshift_values, xH)
            if z_end <= z_min:
                filtered_files.append(file_path)
            else:
                num_excluded += 1
        self.files = filtered_files
        logging.info(f"Excluded {num_excluded} simulations where EoR ends before z = {z_min}.")
        logging.info(f"Remaining simulations after filtering: {len(self.files)}.")

    @staticmethod
    def compute_reionization_end_redshift(redshift_values, xH_values, threshold=0.1):
        idx = np.where(xH_values <= threshold)[0]
        if idx.size > 0:
            z_end = redshift_values[idx[-1]]
        else:
            z_end = redshift_values[-1]
        return z_end

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations and return
        the power spectrum, label, and (if computed) the sample weight.

        Returns:
            ps_tensor (torch.Tensor): Power spectrum array (e.g., shape (3, 10, 10)).
            label_tensor (torch.Tensor): Target label (xH1, xH2, xH3).
            (Optional) weight_tensor (torch.Tensor): Importance weight (scalar).
        """
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract the power spectrum and label.
        ps = data['image']  # Expected shape: (3, 10, 10)
        # Normalize the power spectrum.
        ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)

        label = data['label'][:3]  # xH values (first three entries).
        
        ps_tensor = torch.tensor(ps, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.compute_weights:
            weight_tensor = torch.tensor(self.weights[idx], dtype=torch.float32)
            return ps_tensor, label_tensor, weight_tensor
        else:
            return ps_tensor, label_tensor

##########################################################################



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
    def __init__(self, data_dirs, skip_boundary_labels=False, skip_fraction=1.0, compute_weights=False):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally, compute importance sampling weights.

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            skip_boundary_labels (bool): If True, skip samples with label values close to 0 or 1.
            skip_fraction (float): Fraction of boundary samples to drop.
            compute_weights (bool): If True, precompute importance sampling weights.
        """
        self.data_dirs = data_dirs
        self.skip_boundary_labels = skip_boundary_labels
        self.skip_fraction = skip_fraction
        self.compute_weights = compute_weights
        self.files = []

        # Collect all .npz files from each directory
        for data_dir in self.data_dirs:
            self.files.extend([
                os.path.join(data_dir, f) 
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ])

        logging.info(f"Found {len(self.files)} .npz files across {len(self.data_dirs)} directories.")

        # If weights are requested, precompute a weight for each sample.
        self.weights = None
        if self.compute_weights:
            label_means = []
            for file_path in self.files:
                data = np.load(file_path)
                # Use only the first three label values (xH values).
                label = data['label'][:3]
                # Compute a scalar summary: the mean.
                label_means.append(np.mean(label))
            label_means = np.array(label_means)
            self.weights = compute_sample_weights(label_means)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations and return the power spectrum,
        label, and (optionally) the sample weight.
        """
        orig_idx = idx
        tol = 1e-6  # tolerance for boundary check

        while True:
            file_path = self.files[idx]
            data = np.load(file_path)

            label = data['label'][:3]
            # Skip sample if boundary labels and skipping is enabled.
            if self.skip_boundary_labels and (np.any(np.isclose(label, 0, atol=tol)) or np.any(np.isclose(label, 1, atol=tol))):
                if np.random.rand() < self.skip_fraction:
                    idx = (idx + 1) % len(self.files)
                    if idx == orig_idx:
                        raise IndexError("No valid samples found after boundary filtering.")
                    continue

            ps = data['image']  # Shape (3, 10, 10)
            ps_flattened = ps.reshape(-1)
            # Normalize the power spectrum.
            ps_flattened = (ps_flattened - np.min(ps_flattened)) / (np.max(ps_flattened) - np.min(ps_flattened) + 1e-6)
            # Append redshifts (assumed stored in positions 3:6, normalized by 10).
            redshifts = data['label'][3:6] / 10.0
            ps_flattened = np.concatenate([ps_flattened, redshifts])
        
            ps_tensor = torch.tensor(ps_flattened, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            if self.compute_weights:
                weight = torch.tensor(self.weights[idx], dtype=torch.float32)
                return ps_tensor, label_tensor, weight
            else:
                return ps_tensor, label_tensor





class PowerSpectrumDataset_flow1(Dataset):
    def __init__(self, data_dirs):
        """
        Initialize the dataset with a list of directories containing .npz files.

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
        """
        self.data_dirs = data_dirs
        self.files = []

        # Collect all .npz files from each directory
        for data_dir in self.data_dirs:
            self.files.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])

        # Debug: Print the total number of files found
        logging.info(f"Found {len(self.files)} .npz files in total across {len(self.data_dirs)} directories.")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single .npz file, apply transformations and return
        the power spectrum and the corresponding label.

        Args:
            idx (int): Index of the file to load.

        Returns:
            ps_tensor (torch.Tensor): Flattened and normalized power spectrum.
            label_tensor (torch.Tensor): Target label (xH1, xH2, xH3).
        """
        while True:
            # Load the npz file at the given index
            file_path = self.files[idx]
            data = np.load(file_path)

            # Extract the power spectrum and label
            ps = data['image'] # Shape (3, 10, 10)
            ps = ps[1:3,:,:] # take only last redshift
           
            # Flatten the array
            ps_flattened = ps.reshape(-1)
          
            label = data['label'][1:3]  # Extract only (xH1, xH2, xH3)
      
            ps_flattened = (ps_flattened - np.min(ps_flattened)) / (np.max(ps_flattened) - np.min(ps_flattened) + 1e-6)
           
            #ps_flattened = np.concatenate([ps_flattened]) # append normalised redshifts
        
            # Convert to PyTorch tensors
            ps_tensor = torch.tensor(ps_flattened, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            return ps_tensor, label_tensor



from torch.utils.data import Dataset

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
