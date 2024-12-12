import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging

class PowerSpectrumDataset_global(Dataset):
    def __init__(
        self,
        data_dirs,
        exclude_unfinished_reionization=False,
        xH_threshold=0.01,
        z_target=5., # 5
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

        # Apply normalization to the 3D array
        ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)  # Z-score normalization
        ps = ps / (np.max(np.abs(ps)) + 1e-6)  # Scale to [-1, 1]

        label = data['label']  # xH values (30,)

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
        z_min=10.0
    ):
        """
        Initialize the dataset with a list of directories containing .npz files.
        Optionally exclude simulations where reionization has not finished by z_target
        (i.e., xH > xH_threshold at z = z_target), and/or exclude simulations where
        EoR ends before z_min (i.e., EoR ends at a redshift < z_min).

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            exclude_unfinished_reionization (bool): If True, exclude simulations where xH > xH_threshold at z = z_target.
            xH_threshold (float): Threshold for xH to consider reionization unfinished.
            z_target (float): The redshift at which to check if reionization is finished.
            exclude_early_reionization (bool): If True, exclude simulations where EoR ends before z_min.
            z_min (float): Minimum allowed reionization end redshift. Simulations with EoR ending before this redshift are excluded.
        """
        self.data_dirs = data_dirs
        self.files = []

        # Collect all .npz files from each directory
        for data_dir in self.data_dirs:
            self.files.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])

        # Debug: Print the total number of files found
        logging.info(f"Found {len(self.files)} .npz files in total across {len(self.data_dirs)} directories.")

        # Load redshift values from the first file
        if len(self.files) == 0:
            raise ValueError("No .npz files found in the given directories.")
        first_file = self.files[0]
        self.redshift_values = np.array([6.54, 7.19, 7.96])  # e.g., shape might be (3,)

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
            z_target (float): The redshift at which to check if reionization is finished.
        """
        filtered_files = []
        num_excluded = 0

        # Find the index in redshift_values closest to z_target
        idx_z = np.argmin(np.abs(self.redshift_values - z_target))
        z_actual = self.redshift_values[idx_z]

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label']  # xH values (e.g., (3,) for SKA dataset)
            
            # Ensure ordering of xH is the same as redshift_values
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
            z_min (float): Minimum allowed reionization end redshift.
        """
        filtered_files = []
        num_excluded = 0

        for file_path in self.files:
            data = np.load(file_path)
            xH = data['label']  # e.g., xH shape (3,)

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
        Load a single .npz file, apply transformations and return
        the power spectrum and the corresponding label for CNN.

        Args:
            idx (int): Index of the file to load.

        Returns:
            ps_tensor (torch.Tensor): Power spectrum array (3, 10, 10).
            label_tensor (torch.Tensor): Target label (xH1, xH2, xH3).
        """
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract the power spectrum and label
        ps = data['image']  # Shape (3, 10, 10)

        # Apply normalization
        ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)
        ps = ps / (np.max(np.abs(ps)) + 1e-6)

        label = data['label'][:3]  # Extract only (xH1, xH2, xH3)

        # Convert to PyTorch tensors
        ps_tensor = torch.tensor(ps, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
    
        return ps_tensor, label_tensor



class PowerSpectrumDataset(Dataset):
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

             # Check if the entire power spectrum is zero
            if np.all(ps == 0):
                # Increment the index to skip this sample
                idx = (idx + 1) % len(self.files)
                continue

            # Flatten the array
            ps_flattened = ps.reshape(-1)

            label = data['label'][:3]  # Extract only (xH1, xH2, xH3)
            redshifts = data['label'][3:6] / 10.0 # Extract the redshifts (PS1, PS2, PS3) and normalise
            #global_Tb = data['label'][6:9] 
            #global_Tb = global_Tb / (np.max(abs(global_Tb)) + 1e-6)
    
            #ps_flattened = np.log10(ps_flattened + 1e-6)
        
            ps_flattened = (ps_flattened - np.mean(ps_flattened)) / np.std(ps_flattened + 1e-6)
            ps_flattened = ps_flattened / (np.max(abs(ps_flattened)) + 1e-6)
        

            """
            #ps_flattened = (ps_flattened - np.min(ps_flattened)) / (np.max(ps_flattened) - np.min(ps_flattened) + 1e-6)
            #ps_flattened = 1 / (1 + np.exp(-ps_flattened))
            """
        
            ps_flattened = np.concatenate([ps_flattened, redshifts]) # append normalised redshifts
        
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
