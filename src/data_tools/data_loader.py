import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import h5py
import re

# Default Dataset for EoRFlow, reading directly from .h5 files
class EoRH5Dataset(Dataset):
    def __init__(
        self,
        data_dirs,
        mode='ps2d',               # 'cnn', 'ps1d' or 'ps2d'
        min_redshift_index=0,     # for slicing xHI_labels and ps
        max_redshift_index=15,    # if None uses full length, 15 for z~12
        logit=True,
        add_noise=False,          # add Gaussian noise to PS
        num_files=None,
    ):
        assert mode in ('cnn','ps1d','ps2d', 'skatr'), "mode must be 'cnn', 'ps1d', 'ps2d' or 'skatr'"
        self.mode = mode
        self.logit = logit
        self.add_noise = add_noise
        self.min_z = min_redshift_index
        self.max_z = max_redshift_index

        # Gather all .h5 in data_dirs
        files = []
        for d in data_dirs:
            files += glob.glob(os.path.join(d, '*.h5'))
        files = sorted(files)
        if num_files is not None:
            files = files[:num_files]
        self.files = files
        # read first file to get redshift values
        with h5py.File(files[0],'r') as f:
            ps_group = f['ps']
            self.ps_redshifts = ps_group['ps_redshifts'][...]
            self.lc_redshifts = f['lightcone_redshifts'][...]

        self.z_start = 0
        self.z_end   = 1750 # cut include z=12 for all lightcones

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fn = self.files[idx]
        with h5py.File(fn,'r') as f:
            if self.mode == 'cnn':
                # load 3D image and xHI + optionally redshifts
                img = f['lightcones/brightness_temp'][..., self.z_start:self.z_end].astype('f4')
                lbl = f['xHI_labels'][self.min_z:self.max_z].astype('f4')
                # normalize image
                img = img / 1250
                #img = (img - img.min())/(img.max()-img.min()+1e-6)
                # logit on label?
                if self.logit:
                    eps = 1e-5
                    x = eps + (1-2*eps)*lbl
                    lbl = np.log(x/(1-x))
                # build tensors
                return (
                    torch.tensor(img, dtype=torch.float32),
                    torch.tensor(lbl, dtype=torch.float32),
                )

            else:
                # Power-spectrum modes
                grp = f['ps']
                ps1d = grp['ps1d'][...]  # shape (15,14)
                ps2d = grp['ps2d'][...]  # shape (15,10,10)
                zs   = self.ps_redshifts[self.min_z:self.max_z]
                # Apply redshift slicing to power spectra
                ps1d = ps1d[self.min_z:self.max_z]
                ps2d = ps2d[self.min_z:self.max_z]

                # add gaussian noise?
                if self.add_noise:
                    if self.mode=='ps1d':
                        ps1d += np.random.normal(0,0.05, ps1d.shape)
                    else:
                        ps2d += np.random.normal(0,0.05, ps2d.shape)

                if self.mode=='ps1d':
                    ps = ps1d
                    ps = ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)
                else:
                    ps = ps2d
                    ps = ps = (ps - np.min(ps)) / (np.max(ps) - np.min(ps) + 1e-6)
         
                # flatten per-redshift+append normalized z
                cond = []
                for i, zval in enumerate(zs):
                    slice_i = ps[i].flatten()
                    znorm = (zval - zs.min())/(zs.max()-zs.min()+1e-6)
                    cond.append( np.concatenate([slice_i, [znorm]]) )
                cond = np.stack(cond, axis=0)  # shape (n_slices, feat_dim)
                cond = cond.reshape(-1)        # flatten all together

                # labels: xHI_labels
                lbl = f['xHI_labels'][...].astype('f4')
                # Apply redshift slicing to labels
                lbl = lbl[self.min_z:self.max_z]

                # logit?
                if self.logit:
                    eps=1e-5
                    x=eps+(1-2*eps)*lbl
                    lbl = np.log(x/(1-x))

                return (
                    torch.tensor(cond, dtype=torch.float32),
                    torch.tensor(lbl, dtype=torch.float32),
                )
            

##################################################################
# SKATR embedding dataset loader

class SkatrGridDataset(Dataset):
    """
    Loader for current SKATR embedding outputs.

    Expected format (from SKATR transfer embedding pipeline):
      - one directory (or many) containing lightcone_*.npz
      - each file has:
          image: (D,) summary/embedding vector
          label: parameter vector

    Optional for xHI training:
      - xHI_labels in each npz file (shape (T,))
            - or sidecar labels in separate folders, keyed by simulation id (simuXXXXX)
    """

    SIM_ID_RE = re.compile(r"loreli_(\d+)|simu(\d+)")

    def __init__(
        self,
        data_dirs=None,
        npz_dirs=None,
        target='xhi',
        dtype=torch.float32,
        logit=True,
        min_redshift_index=0,
        max_redshift_index=15,
        sim_param_indices=None,
        drop_tvir=True,
        num_sim_params=5,
        normalize_cond=False,
        cond_normalization=None,
        cond_norm_stats_path=None,
        normalize_sim_params=False,
        sim_param_log10_indices=None,
        target_norm_stats_path=None,
        xhi_labels_dirs=None,
        xhi_label_key='xHI_labels',
        num_files=None,
        cfm_aug=False,
    ):
        # cfm_aug: when True, summaries npz files are expected to carry image of
        # shape (n_aug, D) (the D4 orbit precomputed by embed_dataset_memmap_d4aug.py)
        # and __getitem__ randomly samples one of the n_aug slots per call.
        # When False, image is either (D,) (regular store) OR (n_aug, D) in which
        # case slot 0 (identity) is taken -- so val/test loaders sharing the same
        # summary dir get deterministic outputs.
        self.cfm_aug = bool(cfm_aug)
        # Backward-compatible arg name.
        if data_dirs is None:
            data_dirs = npz_dirs
        if data_dirs is None:
            raise ValueError("Provide data_dirs (or npz_dirs)")
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.target = target
        self.dtype = dtype
        self.logit = logit
        self.min_z = min_redshift_index
        self.max_z = max_redshift_index
        self.sim_param_indices = sim_param_indices
        self.drop_tvir = drop_tvir
        self.num_sim_params = num_sim_params
        self.normalize_cond = normalize_cond
        # Backward-compatible condition normalization behavior:
        # - cond_normalization='zscore' for train-split fitted per-dim normalization
        # - cond_normalization='per_sample_minmax' for legacy behavior
        # - bool normalize_cond=True maps to legacy per-sample minmax
        if cond_normalization is None:
            self.cond_normalization = 'per_sample_minmax' if bool(normalize_cond) else 'none'
        else:
            self.cond_normalization = str(cond_normalization)
        self.normalize_sim_params = normalize_sim_params
        self.sim_param_log10_indices = list(sim_param_log10_indices) if sim_param_log10_indices is not None else None
        self.xhi_label_key = str(xhi_label_key)

        if xhi_labels_dirs is None:
            self.xhi_labels_dirs = []
        elif isinstance(xhi_labels_dirs, str):
            self.xhi_labels_dirs = [xhi_labels_dirs]
        else:
            self.xhi_labels_dirs = list(xhi_labels_dirs)

        # Target normalization stats for sim_params.
        self.target_norm_stats = None
        # Conditioning normalization stats for embeddings.
        self.cond_norm_stats = None
        # Cache sidecar xHI labels to reduce repeated disk I/O.
        self._xhi_cache = {}

        self.files = []
        for d in data_dirs:
            self.files.extend(sorted(glob.glob(os.path.join(d, 'lightcone_*.npz'))))

        if len(self.files) == 0:
            raise ValueError(f"No lightcone_*.npz files found in: {data_dirs}")

        if num_files is not None:
            self.files = self.files[:num_files]

        # Peek first file to infer dimensions and validate keys.
        sample = np.load(self.files[0])
        sample_img = np.asarray(sample['image'])
        # cond_dim is always the LAST axis of image (D for both (D,) and (n_aug, D) layouts).
        # n_aug_slots is the augmentation orbit size (1 for legacy single-summary stores).
        if sample_img.ndim == 1:
            self.n_aug_slots = 1
            self.cond_dim = int(sample_img.shape[0])
        elif sample_img.ndim == 2:
            self.n_aug_slots = int(sample_img.shape[0])
            self.cond_dim = int(sample_img.shape[1])
        else:
            raise ValueError(
                f"Unexpected image ndim={sample_img.ndim} in {self.files[0]}; "
                "expected (D,) for single-summary or (n_aug, D) for D4-augmented stores."
            )
        if self.cfm_aug and self.n_aug_slots == 1:
            raise ValueError(
                "cfm_aug=True but the summary store is a single-summary layout (n_aug_slots=1). "
                f"Point data_dirs at a D4-augmented store. Sample file: {self.files[0]}"
            )

        # Build sidecar lookup map when requested.
        self.xhi_sidecar_map = {}
        if self.xhi_labels_dirs:
            for d in self.xhi_labels_dirs:
                for p in sorted(glob.glob(os.path.join(d, 'simu*.npz'))):
                    sim_id = os.path.splitext(os.path.basename(p))[0]
                    # Keep first occurrence if duplicated across dirs.
                    if sim_id not in self.xhi_sidecar_map:
                        self.xhi_sidecar_map[sim_id] = p

        if self.target == 'xhi':
            has_any_sidecar = bool(self.xhi_sidecar_map)
            if (self.xhi_label_key not in sample.files) and not has_any_sidecar:
                raise ValueError(
                    "target='xhi' requested, but no embedded xHI labels were found in summary files "
                    "and no sidecar xHI labels were configured via xhi_labels_dirs"
                )

            # Keep only files that can provide xHI labels:
            # 1) embedded in summary file, or
            # 2) available in sidecar map by simulation id.
            valid_files = []
            skipped_files = []
            for f in self.files:
                use_embedded = False
                try:
                    d = np.load(f, allow_pickle=False)
                    use_embedded = self.xhi_label_key in d.files
                except Exception:
                    use_embedded = False

                if use_embedded:
                    valid_files.append(f)
                    continue

                sim_id = self._parse_sim_id(f)
                if sim_id is not None and sim_id in self.xhi_sidecar_map:
                    valid_files.append(f)
                else:
                    skipped_files.append(f)

            if len(valid_files) == 0:
                raise ValueError(
                    "target='xhi' requested, but no summaries with valid embedded/sidecar xHI labels were found"
                )

            if skipped_files:
                preview = ', '.join(os.path.basename(m) for m in skipped_files[:5])
                print(
                    f"[SkatrGridDataset] target='xhi': skipping {len(skipped_files)} summaries without labels. "
                    f"Examples: {preview}"
                )
            self.files = valid_files

        if target_norm_stats_path is not None:
            self.load_target_normalization_stats(target_norm_stats_path)
        if cond_norm_stats_path is not None:
            self.load_cond_normalization_stats(cond_norm_stats_path)

    def _select_sim_params(self, label):
        label = np.asarray(label, dtype=np.float32).reshape(-1)

        if self.sim_param_indices is not None:
            return label[np.array(self.sim_param_indices, dtype=int)]

        # Default policy requested for current use case:
        # remove dummy T_vir (first slot), then keep 5 parameters.
        start = 1 if self.drop_tvir else 0
        end = start + int(self.num_sim_params) if self.num_sim_params is not None else len(label)
        end = min(end, len(label))
        selected = label[start:end]

        if selected.size == 0:
            raise ValueError("No simulation parameters selected. Check indices/settings.")
        return selected.astype(np.float32)

    def _parse_sim_id(self, summary_path):
        name = os.path.splitext(os.path.basename(str(summary_path)))[0]
        m = self.SIM_ID_RE.search(name)
        if m is None:
            return None
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        return f"simu{int(raw):05d}"

    def _load_sidecar_xhi(self, summary_path):
        sim_id = self._parse_sim_id(summary_path)
        if sim_id is None:
            raise ValueError(f"Could not parse simulation id from summary filename: {summary_path}")

        if sim_id in self._xhi_cache:
            return self._xhi_cache[sim_id]

        sidecar = self.xhi_sidecar_map.get(sim_id)
        if sidecar is None:
            raise ValueError(
                f"No xHI sidecar label found for {sim_id}. "
                f"Configure xhi_labels_dirs and ensure {sim_id}.npz exists"
            )

        data = np.load(sidecar, allow_pickle=True)
        if self.xhi_label_key not in data.files:
            raise ValueError(
                f"Key '{self.xhi_label_key}' not found in sidecar file {sidecar}. "
                f"Available keys: {list(data.files)}"
            )

        xhi = np.asarray(data[self.xhi_label_key], dtype=np.float32).reshape(-1)
        self._xhi_cache[sim_id] = xhi
        return xhi

    def _select_target(self, data, summary_path=None):
        if self.target == 'sim_params':
            if 'label' not in data:
                raise ValueError("Expected 'label' key for sim_params target")
            y = self._select_sim_params(data['label'])
            if self.target_norm_stats is not None:
                y = self.normalize_targets(y)
            return y

        if self.target == 'xhi':
            if self.xhi_label_key in data.files:
                xhi = np.asarray(data[self.xhi_label_key], dtype=np.float32).reshape(-1)
            else:
                if summary_path is None:
                    raise ValueError("summary_path is required to resolve sidecar xHI labels")
                xhi = self._load_sidecar_xhi(summary_path)
            xhi = xhi[self.min_z:self.max_z]
            if self.logit:
                eps = 1e-5
                x = eps + (1 - 2 * eps) * xhi
                xhi = np.log(x / (1 - x))
            return xhi.astype(np.float32)

        raise ValueError(f"Unknown target type: {self.target}")

    def _default_log10_indices(self, dim):
        # Loreli-friendly default: the first two parameters are strictly positive
        # and span wider dynamic ranges, so log10 stabilizes them.
        if self.sim_param_log10_indices is not None:
            return [int(i) for i in self.sim_param_log10_indices if 0 <= int(i) < dim]
        if dim >= 2:
            return [0, 1]
        return [0] if dim == 1 else []

    def _forward_target_transform(self, y):
        y = np.asarray(y, dtype=np.float32)
        if self.target_norm_stats is None:
            return y
        out = y.copy()
        for idx in self.target_norm_stats.get('log10_indices', []):
            out[..., idx] = np.log10(np.clip(out[..., idx], 1e-12, None))
        mean = np.asarray(self.target_norm_stats['mean'], dtype=np.float32)
        std = np.asarray(self.target_norm_stats['std'], dtype=np.float32)
        return (out - mean) / std

    def _inverse_target_transform(self, y):
        y = np.asarray(y, dtype=np.float32)
        if self.target_norm_stats is None:
            return y
        mean = np.asarray(self.target_norm_stats['mean'], dtype=np.float32)
        std = np.asarray(self.target_norm_stats['std'], dtype=np.float32)
        out = y * std + mean
        for idx in self.target_norm_stats.get('log10_indices', []):
            out[..., idx] = np.power(10.0, out[..., idx])
        return out

    def fit_target_normalization(self, indices=None):
        """Fit sim-parameter normalization statistics from selected samples.

        Intended use: call with training split indices so validation/test remain held out.
        """
        if self.target != 'sim_params':
            return None

        if indices is None:
            indices = range(len(self.files))

        ys = []
        for idx in indices:
            data = np.load(self.files[int(idx)])
            ys.append(self._select_sim_params(data['label']))
        ys = np.asarray(ys, dtype=np.float32)
        if ys.ndim != 2 or ys.shape[0] == 0:
            raise ValueError("Cannot fit target normalization: empty or invalid target array")

        dim = ys.shape[1]
        log10_indices = self._default_log10_indices(dim)
        y_t = ys.copy()
        for idx in log10_indices:
            y_t[:, idx] = np.log10(np.clip(y_t[:, idx], 1e-12, None))

        mean = y_t.mean(axis=0).astype(np.float32)
        std = y_t.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-8)

        self.target_norm_stats = {
            'method': 'zscore',
            'log10_indices': [int(i) for i in log10_indices],
            'mean': mean,
            'std': std,
        }
        self.normalize_sim_params = True
        return self.target_norm_stats

    def set_target_normalization_stats(self, stats):
        if stats is None:
            self.target_norm_stats = None
            return
        self.target_norm_stats = {
            'method': str(stats.get('method', 'zscore')),
            'log10_indices': [int(i) for i in stats.get('log10_indices', [])],
            'mean': np.asarray(stats['mean'], dtype=np.float32),
            'std': np.maximum(np.asarray(stats['std'], dtype=np.float32), 1e-8),
        }
        self.normalize_sim_params = True

    def save_target_normalization_stats(self, path):
        if self.target_norm_stats is None:
            raise ValueError("No target normalization stats available to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            method=self.target_norm_stats.get('method', 'zscore'),
            log10_indices=np.asarray(self.target_norm_stats.get('log10_indices', []), dtype=np.int64),
            mean=np.asarray(self.target_norm_stats['mean'], dtype=np.float32),
            std=np.asarray(self.target_norm_stats['std'], dtype=np.float32),
        )

    def load_target_normalization_stats(self, path):
        stats = np.load(path, allow_pickle=True)
        method_raw = stats['method']
        if np.isscalar(method_raw):
            method = str(method_raw.item()) if hasattr(method_raw, 'item') else str(method_raw)
        else:
            method = str(method_raw)
        loaded = {
            'method': method,
            'log10_indices': np.asarray(stats['log10_indices'], dtype=np.int64).tolist(),
            'mean': np.asarray(stats['mean'], dtype=np.float32),
            'std': np.asarray(stats['std'], dtype=np.float32),
        }
        self.set_target_normalization_stats(loaded)
        return self.target_norm_stats

    def normalize_targets(self, y):
        if self.target_norm_stats is None:
            return np.asarray(y, dtype=np.float32)
        return self._forward_target_transform(y).astype(np.float32)

    def denormalize_targets(self, y):
        if self.target_norm_stats is None:
            return np.asarray(y, dtype=np.float32)
        return self._inverse_target_transform(y).astype(np.float32)

    def fit_cond_normalization(self, indices=None):
        """Fit per-dimension z-score normalization for embedding conditions.

        For D4-augmented stores (image shape (n_aug, D)), stats are fit on the
        identity slot only -- deterministic and matches what val/test consume.
        """
        if indices is None:
            indices = range(len(self.files))

        xs = []
        for idx in indices:
            data = np.load(self.files[int(idx)])
            img = np.asarray(data['image'], dtype=np.float32)
            cond = (img[0] if img.ndim == 2 else img).reshape(-1)
            xs.append(cond)
        xs = np.asarray(xs, dtype=np.float32)
        if xs.ndim != 2 or xs.shape[0] == 0:
            raise ValueError("Cannot fit condition normalization: empty or invalid condition array")

        mean = xs.mean(axis=0).astype(np.float32)
        std = xs.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-8)

        self.cond_norm_stats = {
            'method': 'zscore',
            'mean': mean,
            'std': std,
        }
        self.cond_normalization = 'zscore'
        return self.cond_norm_stats

    def set_cond_normalization_stats(self, stats):
        if stats is None:
            self.cond_norm_stats = None
            return
        self.cond_norm_stats = {
            'method': str(stats.get('method', 'zscore')),
            'mean': np.asarray(stats['mean'], dtype=np.float32),
            'std': np.maximum(np.asarray(stats['std'], dtype=np.float32), 1e-8),
        }
        self.cond_normalization = 'zscore'

    def save_cond_normalization_stats(self, path):
        if self.cond_norm_stats is None:
            raise ValueError("No condition normalization stats available to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            method=self.cond_norm_stats.get('method', 'zscore'),
            mean=np.asarray(self.cond_norm_stats['mean'], dtype=np.float32),
            std=np.asarray(self.cond_norm_stats['std'], dtype=np.float32),
        )

    def load_cond_normalization_stats(self, path):
        stats = np.load(path, allow_pickle=True)
        method_raw = stats['method']
        if np.isscalar(method_raw):
            method = str(method_raw.item()) if hasattr(method_raw, 'item') else str(method_raw)
        else:
            method = str(method_raw)
        loaded = {
            'method': method,
            'mean': np.asarray(stats['mean'], dtype=np.float32),
            'std': np.asarray(stats['std'], dtype=np.float32),
        }
        self.set_cond_normalization_stats(loaded)
        return self.cond_norm_stats

    def normalize_condition(self, cond):
        cond = np.asarray(cond, dtype=np.float32).reshape(-1)
        if self.cond_normalization == 'zscore' and self.cond_norm_stats is not None:
            mean = np.asarray(self.cond_norm_stats['mean'], dtype=np.float32)
            std = np.asarray(self.cond_norm_stats['std'], dtype=np.float32)
            return ((cond - mean) / std).astype(np.float32)
        if self.cond_normalization == 'per_sample_minmax':
            cmin = float(np.min(cond))
            cmax = float(np.max(cond))
            return ((cond - cmin) / (cmax - cmin + 1e-6)).astype(np.float32)
        return cond

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        img = np.asarray(data['image'], dtype=np.float32)
        # Two layouts supported:
        #  - (D,)        : single summary, used by all legacy stores.
        #  - (n_aug, D)  : D4-augmented store (embed_dataset_memmap_d4aug.py).
        #                  cfm_aug=True samples a random slot per call (training);
        #                  cfm_aug=False takes slot 0 = identity (val/test).
        if img.ndim == 2:
            if self.cfm_aug:
                slot = int(np.random.randint(0, img.shape[0]))
            else:
                slot = 0
            cond = img[slot].reshape(-1)
        else:
            cond = img.reshape(-1)
        cond = self.normalize_condition(cond)

        y = self._select_target(data, summary_path=self.files[idx])

        c = torch.tensor(cond, dtype=self.dtype)
        y = torch.tensor(y, dtype=self.dtype)
        return c, y
