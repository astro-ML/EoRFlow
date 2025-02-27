import os
import glob
import logging
import math
import numpy as np
import py21cmfast as p21c
from multiprocessing import Pool
import multiprocessing as mp

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
    Class to read LightCone objects from .h5 files, add noise to brightness_temp,
    and save outputs as .npz files (similar to your old workflow).
    """

    def __init__(
        self, 
        source_dir: str,
        destination_dir: str,
        noise_level: str = "opt"
    ):
        """
        Args:
            source_dir (str): Directory containing .h5 LightCone files.
            destination_dir (str): Directory to save noisy .npz files.
            noise_level (str): Placeholder parameter to choose noise variant ("opt" or "mod").
        """
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.noise_level = noise_level

        if not os.path.exists(self.destination_dir):
            os.makedirs(self.destination_dir)

    def read_lightcone(self, filename: str):
        """
        Read a .h5 LightCone file using 21cmFAST and extract relevant quantities.

        Returns:
            brightness_temp (np.ndarray)
            params (np.ndarray)
            lc_redshifts (np.ndarray)
            gxH (np.ndarray)
            gxH_redshifts (np.ndarray)
        """
        logging.info(f"Reading LightCone from {filename}")
        lightcone = p21c.outputs.LightCone.read(filename)

        OM = lightcone.cosmo_params.OMm
        L_X = lightcone.astro_params.L_X
        T_vir = lightcone.astro_params.ION_Tvir_MIN
        E0 = lightcone.astro_params.NU_X_THRESH
        Zeta = lightcone.astro_params.HII_EFF_FACTOR
        mDM = 2  # fixed param

        brightness_temp = lightcone.brightness_temp
        lc_redshifts = lightcone.lightcone_redshifts
        gxH = np.flip(lightcone.global_xH)
        gxH_redshifts = np.flip(lightcone.node_redshifts)

        # Parameter array: [mDM, OM, E0, L_X, T_vir, Zeta]
        params = np.array([mDM, OM, E0, L_X, T_vir, Zeta], dtype=np.float32)

        return brightness_temp, params, lc_redshifts, gxH, gxH_redshifts

    def read_noise_files(self):
        """
        Read noise files based on the noise level specified.
        Modify this logic to match your actual noise .npz files.
        """
        if self.noise_level == "opt":
            files = glob.glob(
                "/remote/gpu01a/pietschke/21cm_pie/21cm_pie/generate_data/calcfiles/"
                "opt_mocks/SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz"
            )
        elif self.noise_level == "mod":
            files = glob.glob(
                "/remote/gpu01a/pietschke/21cm_pie/21cm_pie/generate_data/calcfiles/"
                "mod_mocks/SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz"
            )
        else:
            logging.error("Please choose a valid noise level ('opt' or 'mod').")
            raise ValueError("Invalid noise level. Use 'opt' or 'mod'.")
        files.sort(reverse=True)
        return files

    def add_noise(self, brightness_temp: np.ndarray, parameters: np.array) -> np.ndarray:
        """
        Add noise to the simulation using your advanced Fourier approach.
        """
        logging.info('Creating mock with Fourier-based noise...')

        # 1) Read redshifts
        with open("/remote/gpu01a/pietschke/21cm_pie/21cm_pie/generate_data/redshifts5.npy", "rb") as data:
            box_redshifts = np.load(data, allow_pickle=True)
            box_redshifts = sorted(box_redshifts)

        cosmo_params = p21c.CosmoParams(OMm=parameters[1])
        astro_params = p21c.AstroParams(INHOMO_RECO=True)
        user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
        flag_options = p21c.FlagOptions()
        sim_lightcone = p21c.LightCone(
            5., user_params, cosmo_params, astro_params, 
            flag_options, 0, {"brightness_temp": brightness_temp}, 35.05
        )

        redshifts = sim_lightcone.lightcone_redshifts
        cell_size = user_params.BOX_LEN / user_params.HII_DIM
        hii_dim = user_params.HII_DIM

        # 3) Figure out how to split the lightcone by redshift bins
        box_len = []
        y = 0
        z = 0
        n_slices = brightness_temp.shape[-1]
        for x in range(n_slices):
            if (y + 1 < len(box_redshifts) and 
                redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2.0):
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
        noise_files = self.read_noise_files()

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

    def save_npz(self,
                 filename: str,
                 brightness_temp: np.ndarray,
                 params: np.ndarray,
                 lc_redshifts: np.ndarray,
                 gxH: np.ndarray,
                 gxH_redshifts: np.ndarray) -> None:
        """
        Save the final arrays to a .npz file.
        """
        logging.info(f"Saving noisy lightcone to {filename}")
        np.savez(
            filename,
            image=brightness_temp,
            label=params,               
            lc_redshifts=lc_redshifts,
            gxH=gxH,
            gxH_redshifts=gxH_redshifts
        )

    def _process_one_file(self, f):
        """
        Single-file processing function to be used by Pool workers.
        """
        try:
            # Determine output name
            base = os.path.splitext(os.path.basename(f))[0]
            outname = os.path.join(self.destination_dir, base + "_noisy.npz")

            # If output already exists, skip
            if os.path.exists(outname):
                logging.info(f"[SKIP] {outname} already exists. Skipping.")
                return

            # Otherwise proceed
            logging.info(f"Processing file: {f}")
            brightness_temp, params, lc_redshifts, gxH, gxH_redshifts = self.read_lightcone(f)
            mock_bt = self.add_noise(brightness_temp, params)
            self.save_npz(outname, mock_bt, params, lc_redshifts, gxH, gxH_redshifts)
            logging.info(f"[DONE] Finished processing {f}")

        except Exception as e:
            logging.error(f"Error processing file {f}: {e}")

    def convert_lightcones(self, num_files='all', n_cores=4) -> None:
        """
        Main loop: 
        1) Finds .h5 files in source_dir.
        2) Optionally limit to 'num_files' (int or 'all').
        3) Uses multiprocessing.Pool with n_cores to process each file in parallel.
        """
        h5_files = sorted(glob.glob(os.path.join(self.source_dir, "*.h5")))
        if not h5_files:
            logging.warning(f"No .h5 files found in {self.source_dir}!")
            return

        if num_files != 'all':
            h5_files = h5_files[:num_files]

        logging.info(f"Converting {len(h5_files)} LightCone file(s) from {self.source_dir} using {n_cores} cores.")

        # Create a pool of worker processes
        with Pool(processes=n_cores) as pool:
            pool.map(self._process_one_file, h5_files)

        logging.info("All parallel tasks have finished.")

def main():
    # ---------------------------
    #  Configuration
    # ---------------------------
    source_dir      = "/remote/gpu01a/schlenker/21cm-wrapper/data_sbi"
    destination_dir = "/remote/gpu01a/pietschke/lightcones/mock_tom"
    
    noise_level = "opt"         # or "mod"
    num_files   = "all"         # process all .h5
    n_cores     = 8             # how many parallel processes to use

    converter = NoiseH5Converter(source_dir, destination_dir, noise_level=noise_level)
    converter.convert_lightcones(num_files=num_files, n_cores=n_cores)

if __name__ == "__main__":
    main()