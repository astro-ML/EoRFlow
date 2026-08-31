import os
import glob
import logging
import math
import h5py
import numpy as np
import py21cmfast as p21c

# MPI rank/size from OpenMPI env
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

# Set up logging
log_filename = 'add_noise.log'
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class NoiseH5Converter:
    """
    Read LightCone .h5 files, add Fourier-based noise to brightness_temp,
    and save updated cubes as .h5 with the same structure.
    Includes AA4 opt, mod, and aaStar_mod noise levels.
    """

    def __init__(self, source_dir, destination_dir, noise_level="opt", start=0, n_local=None):
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.noise_level = noise_level
        self.start = start
        self.n_local = n_local

        os.makedirs(str(self.destination_dir), exist_ok=True)

    def _read_lightcone(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            cube = f['lightcones/brightness_temp'][...]

            lc_redshifts = f['lightcone_redshifts'][...]

            node_redshifts = f['node_redshifts'][...]

            up = f['user_params'].attrs
            BOX_LEN = up['BOX_LEN']    # comoving Mpc
            HII_DIM = up['HII_DIM']    # should be 140

            cell_size = BOX_LEN / HII_DIM

        return cube, lc_redshifts, node_redshifts, cell_size, HII_DIM

    def _read_noise_files(self):
        """
        Return a sorted list of .npz files for the given noise_level.
        """
        if self.noise_level == "opt":
            pattern = (
                "/pfs/10/work/hd_pt254-eorflow/21cm_pie/twentyone_cm_pie/generate_data/calcfiles/opt_mocks/"
                "SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz"
            )
        elif self.noise_level == "mod":
            pattern = (
                "/pfs/10/work/hd_pt254-eorflow/21cm_pie/twentyone_cm_pie/generate_data/calcfiles/mod_mocks/"
                "SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz"
            )
        elif self.noise_level == "aaStar_mod":
            pattern = (
                "/pfs/10/work/hd_pt254-eorflow/21cmSense_extended/mod_noise/SKA1_Low.aastar.track_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz"
            )
        else:
            logging.error("Invalid noise_level '%s'", self.noise_level)
            raise ValueError("noise_level must be 'opt' or 'mod'")

        files = sorted(glob.glob(pattern), reverse=True) # sort by z
        logging.info("Found %d noise files for level '%s'", len(files), self.noise_level)
        return files

    def _add_noise(self, brightness_temp: np.ndarray,
                   lc_redshifts, node_redshifts,
                   cell_size, hii_dim) -> np.ndarray:
        """Add noise via Fourier-domain injection."""
        logging.info("Adding Fourier-based noise")
        # compute box_len per redshift bin
        n_slices = brightness_temp.shape[-1]
        #print(f"Redshift range: {lc_redshifts.min()} to {lc_redshifts.max()}")
        box_redshifts = np.sort(node_redshifts)
        #print(f"Box redshifts: {box_redshifts}")
        # 3) Figure out how to split the lightcone by redshift bins
        box_len = []
        y = 0
        z = 0
        n_slices = brightness_temp.shape[-1]
        for x in range(n_slices):
            if (y + 1 < len(box_redshifts) and 
                lc_redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2.0):
                box_len.append(x - z)
                y += 1
                z = x
        box_len.append(n_slices - z)

        # 4) Split brightness_temp accordingly
        split_lc = []
        idx_start = 0
        for blen in box_len:
            idx_end = idx_start + int(blen)
            split_lc.append(brightness_temp[:, :, idx_start:idx_end])
            idx_start = idx_end

        mock_lc = np.zeros_like(brightness_temp)
        k_1d = np.fft.fftfreq(hii_dim, d=(cell_size / (2.0 * math.pi)))
        noise_files = self._read_noise_files()

        total_slices_done = 0
        for chunk_idx, blen in enumerate(box_len):
            if chunk_idx >= len(noise_files):
                logging.warning("Not enough noise files for this chunk. Will reuse last or skip.")
                break

            with np.load(noise_files[chunk_idx]) as data:
                ks = data["ks"]
                T_errs = data["T_errs"]

            kbox = np.fft.rfftfreq(int(blen), d=(cell_size/(2.0*math.pi)))
            volume = (hii_dim * hii_dim * blen) * (cell_size ** 3)

            chunk_lc = split_lc[chunk_idx]
            deldel_T = np.fft.rfftn(chunk_lc, s=(hii_dim, hii_dim, blen))

            err21a = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(blen)))
            err21b = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(blen)))

            deldel_T_noise = np.zeros_like(deldel_T, dtype=np.complex128)
            deldel_T_mock  = np.zeros_like(deldel_T, dtype=np.complex128)

            for ix in range(hii_dim):
                for iy in range(hii_dim):
                    for iz in range(int(blen//2 + 1)):
                        k_mag = math.sqrt(k_1d[ix]**2 + k_1d[iy]**2 + kbox[iz]**2)
                        err21 = np.interp(k_mag, ks, T_errs)

                        if k_mag != 0:
                            amplitude = math.sqrt(math.pi * math.pi * volume / (k_mag**3) * err21)
                            deldel_T_noise[ix, iy, iz] = amplitude * (err21a[ix, iy, iz] + 1j * err21b[ix, iy, iz])
                        else:
                            deldel_T_noise[ix, iy, iz] = 0.0

                        if err21 >= 1000:
                            deldel_T_mock[ix, iy, iz] = 0.0
                        else:
                            deldel_T_mock[ix, iy, iz] = deldel_T[ix, iy, iz] + deldel_T_noise[ix, iy, iz] / (cell_size**3)

            delta_T_mock = np.fft.irfftn(deldel_T_mock, s=(hii_dim, hii_dim, blen))

            start = total_slices_done
            end   = start + delta_T_mock.shape[-1]
            mock_lc[:, :, start:end] = delta_T_mock
            total_slices_done = end
            logging.info(f"Processed chunk {chunk_idx+1}/{len(box_len)}, slices: {end}/{n_slices}")

        return mock_lc

    def process_one(self, infile: str):
        """Read one .h5, add noise, and write to destination."""
        base   = os.path.basename(infile)
        outpath = os.path.join(self.destination_dir, base)

        if os.path.exists(outpath):
            logging.info("Skipping %s (already exists)", base)
            return

        try:
            # 1) read in the raw cube + extras
            cube, lc_z, node_z, cell, dim = self._read_lightcone(infile)

            # 2) add noise
            bt_noisy = self._add_noise(cube, lc_z, node_z, cell, dim)

            # 3) copy everything but override the lightcones/brightness_temp
            logging.info("Writing noisy HDF5 to %s", outpath)
            with h5py.File(infile, 'r') as src, h5py.File(outpath, 'w') as dst:
                def _copy(name, obj):
                    # 1) If it’s a group, recreate it and copy its attrs
                    if isinstance(obj, h5py.Group):
                        grp = dst.create_group(name)
                        for k, v in obj.attrs.items():
                            grp.attrs[k] = v

                    # 2) If it’s the brightness_temp dataset, overwrite with noisy data
                    elif isinstance(obj, h5py.Dataset) and name == 'lightcones/brightness_temp':
                        ds = dst.create_dataset(
                            name,
                            data=bt_noisy,
                            chunks=(dim, dim, 1),
                            compression='gzip',
                            dtype=bt_noisy.dtype
                        )

                    # 3) Otherwise it’s some other dataset—copy data, dtype, compression, and attrs
                    elif isinstance(obj, h5py.Dataset):
                        ds = dst.create_dataset(
                            name,
                            data=obj[()],
                            dtype=obj.dtype,
                            compression=obj.compression
                        )
                        for k, v in obj.attrs.items():
                            ds.attrs[k] = v

                # Walk through every object in the source file
                src.visititems(_copy)

                # Finally, copy top-level (file) attributes
                for k, v in src.attrs.items():
                    dst.attrs[k] = v

            logging.info("Finished %s", base)

        except Exception as e:
            logging.error("Error processing %s: %s", base, e)

    def convert_all(self):
        all_files = sorted(glob.glob(os.path.join(self.source_dir, '*.h5')))
        # slice for this rank
        if self.n_local is None:
            files = all_files[self.start::size]
        else:
            files = all_files[self.start:self.start+self.n_local]
        logging.info(f"Rank {rank}: converting {len(files)} files (start={self.start})")
        for f in files:
            self.process_one(f)

        logging.info("All done.")

if __name__=='__main__':
    # compute chunking exactly as in your sim script
    source_dir = '/pfs/10/work/hd_pt254-eorflow/database/pure/batch_test'
    TOTAL = len(glob.glob(f"{source_dir}/*.h5"))
    base, rem = divmod(TOTAL, size)
    n_local = base + (1 if rank < rem else 0)
    start = rank*base + min(rank, rem)

    converter = NoiseH5Converter(
        source_dir=source_dir,
        destination_dir=f'/pfs/10/work/hd_pt254-eorflow/database/aaStar_mod_noise/batch_test',
        noise_level='aaStar_mod',
        start=start,
        n_local=n_local
    )
    converter.convert_all()

