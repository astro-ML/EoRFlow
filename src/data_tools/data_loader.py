import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

# original dataloader for power spectra
class PowerSpectrumDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        mode='ps2d',  # 'ps2d' or 'ps1d'
        min_redshift_index=0,
        max_redshift_index=15,
        max_zeros_allowed=15,
        max_ones_allowed=15,
        use_cnn=False,
        logit=False,
        add_noise=False,
        noise_level=0.05,
        aa4_mod_noise=False,
        aaStar_mod_noise=False,
        num_files=None,
    ):
        """
        Dataset for loading power spectrum data, supporting both 1D and 2D power spectra.

        Args:
            data_dirs (list): List of paths to directories containing .npz files.
            mode (str): Type of power spectrum to load ('ps2d' or 'ps1d').
            min_redshift_index (int): Minimum redshift index to include.
            max_redshift_index (int): Maximum redshift index to include.
            max_zeros_allowed (int): Maximum number of 0 values allowed in xH.
            max_ones_allowed (int): Maximum number of 1 values allowed in xH.
            use_cnn (bool): Format the PS for CNN input.
            logit (bool): Apply logit transformation to labels.
            add_noise (bool): Add Gaussian noise to the PS.
            num_files (int): Maximum number of files to load.
        """
        assert mode in ['ps1d', 'ps2d'], "mode must be 'ps1d' or 'ps2d'"
        self.mode = mode
        self.data_dirs = data_dirs
        self.min_redshift_index = min_redshift_index
        self.max_redshift_index = max_redshift_index
        self.max_zeros_allowed = max_zeros_allowed
        self.max_ones_allowed = max_ones_allowed
        self.use_cnn = use_cnn
        self.logit = logit
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.aa4_mod_noise = aa4_mod_noise
        self.aaStar_mod_noise = aaStar_mod_noise
        self.num_files = num_files

        all_files = []
        for data_dir in self.data_dirs:
            all_files.extend([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir) if f.endswith('.npz')
            ])

        logging.info(f"Found {len(all_files)} .npz files across {len(self.data_dirs)} directories.")

        if aa4_mod_noise and aaStar_mod_noise:
            raise ValueError("Cannot use both aa4_mod_noise and aaStar_mod_noise at the same time.")
        elif aa4_mod_noise:
            noise_type = "aa4"
        elif aaStar_mod_noise:
            noise_type = "aaStar"
        else:
            noise_type = None

        if noise_type is not None:
            base = "/remote/gpu01a/pietschke/EoRFlow/data/power_spectra"
            paths = {
                "aa4":   (f"{base}/mean_aa4_mod_noise.npy",
                        f"{base}/std_aa4_mod_noise.npy"),
                "aaStar":(f"{base}/mean_aaStar_mod_noise.npy",
                        f"{base}/std_aaStar_mod_noise.npy"),
            }
            mean_path, std_path = paths[noise_type]
            self.noise_mean = np.load(mean_path)
            self.noise_std  = np.load(std_path)
        else:
            # neither noise flag set
            self.noise_mean = None
            self.noise_std  = None

        first_file = all_files[0]
        data = np.load(first_file)
        self.redshift_values = data['redshifts'][min_redshift_index:max_redshift_index]

        filtered_files = all_files

        if self.num_files is not None:
            filtered_files = filtered_files[-self.num_files:]
            logging.info(f"Limiting the dataset to the last {self.num_files} files.")

        self.files = filtered_files
        logging.info(f"Keeping {len(self.files)} files after all filtering steps.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)

        ps = data[self.mode][self.min_redshift_index:self.max_redshift_index]

        # add moderate noise if set
        if self.noise_mean is not None and self.noise_std is not None and self.mode == 'ps2d':
            noise = np.random.normal(self.noise_mean, self.noise_std, ps.shape)
            ps += noise

        # add gaussian noise if set
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, ps.shape)
            ps += noise

        # normalize
        ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)

        # load labels and apply logit
        label = data['label'][self.min_redshift_index:self.max_redshift_index]
        if self.logit:
            epsilon = 1e-5
            label = epsilon + (1 - 2 * epsilon) * label
            label = np.log(label / (1 - label))

        #load redshifts and normalize
        redshifts = data['redshifts'][self.min_redshift_index:self.max_redshift_index]
        redshifts = (redshifts - np.min(redshifts)) / (np.max(redshifts) - np.min(redshifts) + 1e-6)

        if not self.use_cnn:
            cond = []
            for i in range(len(redshifts)):
                ps_slice = ps[i].flatten()
                z_val = redshifts[np.newaxis, i]
                cond.append(np.concatenate([ps_slice, z_val]))
            condition = np.concatenate(cond)
            condition_tensor = torch.tensor(condition, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return condition_tensor, label_tensor
        else:
            ps_tensor = torch.tensor(ps, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            redshifts_tensor = torch.tensor(redshifts, dtype=torch.float32)
            return ps_tensor, label_tensor, redshifts_tensor
        




# new dataloader for cnn mode, to be finalized
class EoRFlowDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        mode='ps2d',    # 'ps2d', 'ps1d' or 'cnn'
        image_dirs=None,  # only used when mode='cnn'
        min_redshift_index=0,
        max_redshift_index=15,
        max_zeros_allowed=15,
        max_ones_allowed=15,
        filter_reionization_timing=False,
        logit=False,
        add_noise=False,
        noise_level=0.1,
        num_files=None,
        aa4_mod_noise=False,
        aaStar_mod_noise=False,
    ):
        """
        Dataset for loading power spectrum or image+label pairs.
        Args:
            data_dirs (list): dirs containing .npz files for PS or labels (when cnn).
            image_dirs (list): dirs containing .npz image files (only for cnn).
            mode (str): 'ps2d', 'ps1d', or 'cnn'.
        """
        assert mode in ['ps2d', 'ps1d', 'cnn'], "mode must be 'ps2d', 'ps1d', or 'cnn'"
        self.mode = mode
        self.data_dirs = data_dirs
        self.image_dirs = image_dirs
        self.min_redshift_index = min_redshift_index
        self.max_redshift_index = max_redshift_index
        self.max_zeros_allowed = max_zeros_allowed
        self.max_ones_allowed = max_ones_allowed
        self.filter_reionization_timing = filter_reionization_timing
        self.logit = logit
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.num_files = num_files
        self.aa4_mod_noise = aa4_mod_noise
        self.aaStar_mod_noise = aaStar_mod_noise

        # inside your __init__, after validating that both aren't True
        if aa4_mod_noise and aaStar_mod_noise:
            raise ValueError("Cannot use both aa4_mod_noise and aaStar_mod_noise at the same time.")
        elif aa4_mod_noise:
            noise_type = "aa4"
        elif aaStar_mod_noise:
            noise_type = "aaStar"
        else:
            noise_type = None

        if noise_type is not None:
            base = "/remote/gpu01a/pietschke/EoRFlow/data/power_spectra"
            paths = {
                "aa4":   (f"{base}/mean_aa4_mod_noise.npy",
                        f"{base}/std_aa4_mod_noise.npy"),
                "aaStar":(f"{base}/mean_aaStar_mod_noise.npy",
                        f"{base}/std_aaStar_mod_noise.npy"),
            }
            mean_path, std_path = paths[noise_type]
            self.noise_mean = np.load(mean_path)
            self.noise_std  = np.load(std_path)
        else:
            # neither noise flag set
            self.noise_mean = None
            self.noise_std  = None

        if self.mode == 'cnn':
            assert image_dirs is not None, "image_dirs must be provided when mode='cnn'"
            # collect image and label files
            img_files = []
            for d in self.image_dirs:
                img_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz')]
            lbl_files = []
            for d in self.data_dirs:
                lbl_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz')]

            # map basenames to paths
            img_map = {os.path.basename(f): f for f in img_files}
            lbl_map = {os.path.basename(f): f for f in lbl_files}
            # intersect
            common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
            paired = [(img_map[n], lbl_map[n]) for n in common]
            if self.num_files is not None:
                paired = paired[-self.num_files:]
                logging.info(f"Limiting to last {self.num_files} image/label pairs.")
            self.files = paired
            logging.info(f"Found {len(self.files)} matching image/label pairs for CNN mode.")

        else:
            # previous PS handling
            all_files = []
            for data_dir in self.data_dirs:
                all_files.extend([
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir) if f.endswith('.npz')
                ])
            logging.info(f"Found {len(all_files)} .npz files across {len(self.data_dirs)} directories.")

            # load redshift values from first file
            data = np.load(all_files[0])
            self.redshift_values = data['redshifts'][min_redshift_index:max_redshift_index]

            filtered = all_files #self._filter_by_extreme_values(all_files)
            if filter_reionization_timing:
                filtered = self._filter_by_reionization_timing(filtered)
            if num_files is not None:
                filtered = filtered[-num_files:]
                logging.info(f"Limiting to last {num_files} files.")
            self.files = filtered
            logging.info(f"Keeping {len(self.files)} files after filtering.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode == 'cnn':
            img_path, lbl_path = self.files[idx]
            img_data = np.load(img_path)['image']
            label_data = np.load(lbl_path)['label']

            # optionally slice by redshift range if image has time axis
            #try:
            #    img = img_data[self.min_redshift_index:self.max_redshift_index]
            #except Exception:
            img = img_data

            # normalize image to [0,1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            lbl = label_data[self.min_redshift_index:self.max_redshift_index]
            if self.logit:
                eps = 1e-5
                lbl = eps + (1 - 2*eps)*lbl
                lbl = np.log(lbl / (1-lbl))

            # redshifts
            zs = np.load(lbl_path)['redshifts'][self.min_redshift_index:self.max_redshift_index]
            zs = (zs - zs.min()) / (zs.max() - zs.min() + 1e-6)

            img_tensor = torch.tensor(img, dtype=torch.float32)
            lbl_tensor = torch.tensor(lbl, dtype=torch.float32)
            zs_tensor = torch.tensor(zs, dtype=torch.float32)
            return img_tensor, lbl_tensor, zs_tensor

        # PS modes
        file_path = self.files[idx]
        data = np.load(file_path)
        ps = data[self.mode][self.min_redshift_index:self.max_redshift_index]

        # add moderate noise if set
        if self.noise_mean is not None and self.noise_std is not None and self.mode == 'ps2d':
            noise = np.random.normal(self.noise_mean, self.noise_std, ps.shape)
            ps += noise

        # add gaussian noise if set
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, ps.shape)
            print(noise.max(), noise.min())
            ps += noise

        # normalize
        ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)

        # load labels and apply logit
        label = data['label'][self.min_redshift_index:self.max_redshift_index]
        if self.logit:
            eps = 1e-5
            label = eps + (1-2*eps)*label
            label = np.log(label / (1-label))

        # load redshifts and normalize
        zs = data['redshifts'][self.min_redshift_index:self.max_redshift_index]
        zs = (zs - np.min(zs)) / (np.max(zs) - np.min(zs) + 1e-6)

        if self.mode in ['ps1d', 'ps2d']:
            cond = []
            for i in range(len(zs)):
                ps_slice = ps[i].flatten()
                z_val = zs[np.newaxis, i]
                cond.append(np.concatenate([ps_slice, z_val]))
            cond = np.concatenate(cond)
            return torch.tensor(cond, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)



