import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging

class PowerSpectrumDataset_CNN(Dataset):
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
        the power spectrum and the corresponding label for CNN.

        Args:
            idx (int): Index of the file to load.

        Returns:
            ps_tensor (torch.Tensor): 3D power spectrum array (3, 10, 10).
            label_tensor (torch.Tensor): Target label (xH1, xH2, xH3).
        """
        while True:
            # Load the npz file at the given index
            file_path = self.files[idx]
            data = np.load(file_path)

            # Extract the power spectrum and label
            ps = data['image']  # Shape (3, 10, 10)

            # Check if the entire power spectrum is zero
            if np.all(ps == 0):
                # Increment the index to skip this sample
                idx = (idx + 1) % len(self.files)
                continue

            # Apply normalization to the 3D array (same as before but without flattening)
            ps = (ps - np.mean(ps)) / (np.std(ps) + 1e-6)  # Z-score normalization
            ps = ps / (np.max(np.abs(ps)) + 1e-6)  # Scale to [-1, 1]


            label = data['label'][:3]  # Extract only (xH1, xH2, xH3)

            # Convert to PyTorch tensors
            ps_tensor = torch.tensor(ps, dtype=torch.float32)  # Retain (3, 10, 10) shape for CNN
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
        
            #ps_flattened = np.concatenate([ps_flattened, redshifts]) # append normalised redshifts
        
            # Convert to PyTorch tensors
            ps_tensor = torch.tensor(ps_flattened, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            return ps_tensor, label_tensor