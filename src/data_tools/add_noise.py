import os
import glob
import sys
import logging
import math
import numpy as np
from typing import Tuple
import multiprocessing
import py21cmfast as p21c

# Set up logging
log_filename = 'add_noise.log'
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def process_file(args):
    """
    Helper function to process a single file.

    Args:
        args (tuple): A tuple containing the Noise instance, file path, and destination path.

    Returns:
        None
    """
    noise_instance, file, destination_path = args
    try:
        brightness_temp, label, lc_redshifts, gxH, gxH_redshifts = noise_instance.read_lightcones(file)
        mock_brightness_temp = noise_instance.add_noise(brightness_temp, label)
        save_name = os.path.basename(file)
        noise_instance.save(os.path.join(destination_path, save_name), mock_brightness_temp, label, lc_redshifts, gxH, gxH_redshifts)
        logging.info(f'Processed {save_name}')
    except Exception as e:
        logging.error(f'Error processing {file}: {e}')

class Noise:
    """
    Class to generate noise and add to simulated lightcones.
    """
    def __init__(self):
        self.params = {
            'noise': {
                'level': 'opt'  # or 'mod', depending on which noise level you want
            },
            'convert': {
                'do_it': True,
                'source': '/remote/gpu01a/heneka/21cmlightcones/pure_simulations_astro/',
                'n_sims': 'all'  # or specify an integer
            },
            'destination_noise': '/remote/gpu01a/pietschke/lightcones/mock_astro/'
        }

    def read_noise_files(self) -> list:
        """
        Read noise files based on the noise level specified in the parameters.

        Returns:
            list: List of filenames for the noise data.
        """
        noise_level = self.params['noise']['level']
        if noise_level == "opt":
            files = glob.glob("/remote/gpu01a/pietschke/21cm_pie/21cm_pie/generate_data/calcfiles/opt_mocks/SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        elif noise_level == "mod":
            files = glob.glob("/remote/gpu01a/pietschke/21cm_pie/21cm_pie/generate_data/calcfiles/mod_mocks/SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        else:
            logging.info("Please choose a valid foreground model")
            sys.exit()
        files.sort(reverse=True)
        return files

    def add_noise(self, brightness_temp: np.ndarray, parameters: np.array) -> np.ndarray:
        """
        Add noise to the simulation.

        Args:
            brightness_temp (np.ndarray): The brightness temperature data.
            parameters (np.array): The simulation parameters.

        Returns:
            np.ndarray: The brightness temperature data with added noise.
        """
        logging.info('Creating mock lightcone with noise')

        # just for Benedikt
        with open("/remote/gpu01a/pietschke/21cm_pie/21cm_pie/generate_data/redshifts5.npy", "rb") as data:
            box_redshifts = list(np.load(data, allow_pickle=True))
            box_redshifts.sort()
        cosmo_params = p21c.CosmoParams(OMm=parameters[0])
        astro_params = p21c.AstroParams(INHOMO_RECO=True)
        user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
        flag_options = p21c.FlagOptions()
        sim_lightcone = p21c.LightCone(5.0, user_params, cosmo_params, astro_params, flag_options, 0,
                                       {"brightness_temp": brightness_temp}, 35.05)

        # here plug in redshifts
        redshifts = sim_lightcone.lightcone_redshifts
        box_len = np.array([])
        y = 0
        z = 0
        for x in range(len(brightness_temp[0][0])):
            if y + 1 < len(box_redshifts) and redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2:
                box_len = np.append(box_len, x - z)
                y += 1
                z = x
        box_len = np.append(box_len, x - z + 1)
        y = 0
        delta_T_split = []
        for x in box_len:
            delta_T_split.append(brightness_temp[:, :, int(y):int(x + y)])
            y += x

        mock_lc = np.zeros(brightness_temp.shape)
        cell_size = 200 / 140
        hii_dim = 140
        k140 = np.fft.fftfreq(140, d=cell_size / (2 * np.pi))
        index1 = 0
        index2 = 0
        files = self.read_noise_files()
        for idx, x in enumerate(box_len):
            with np.load(files[idx]) as data:
                ks = data["ks"]
                T_errs = data["T_errs"]
            kbox = np.fft.rfftfreq(int(x), d=cell_size / (2 * np.pi))
            volume = hii_dim * hii_dim * x * cell_size ** 3
            err21a = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(x // 2 + 1)))
            err21b = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(x // 2 + 1)))
            deldel_T = np.fft.rfftn(delta_T_split[idx], s=(hii_dim, hii_dim, int(x)))
            deldel_T_noise = np.zeros((hii_dim, hii_dim, int(x // 2 + 1)), dtype=np.complex_)
            deldel_T_mock = np.zeros((hii_dim, hii_dim, int(x // 2 + 1)), dtype=np.complex_)

            for n_x in range(hii_dim):
                for n_y in range(hii_dim):
                    for n_z in range(int(x // 2 + 1)):
                        k_mag = np.sqrt(k140[n_x] ** 2 + k140[n_y] ** 2 + kbox[n_z] ** 2)
                        err21 = np.interp(k_mag, ks, T_errs)

                        if k_mag != 0 and err21 < 1000:
                            factor = np.sqrt((np.pi ** 2 * volume) / (k_mag ** 3) * err21)
                            noise_real = factor * err21a[n_x, n_y, n_z]
                            noise_imag = factor * err21b[n_x, n_y, n_z]
                            deldel_T_noise[n_x, n_y, n_z] = (noise_real + 1j * noise_imag)
                            deldel_T_mock[n_x, n_y, n_z] = deldel_T[n_x, n_y, n_z] + deldel_T_noise[n_x, n_y, n_z] / cell_size ** 3
                        else:
                            deldel_T_mock[n_x, n_y, n_z] = deldel_T[n_x, n_y, n_z]

            delta_T_mock = np.fft.irfftn(deldel_T_mock, s=(hii_dim, hii_dim, int(x)))
            index1 = index2
            index2 += delta_T_mock.shape[2]
            mock_lc[:, :, index1:index2] = delta_T_mock

        return mock_lc

    def read_lightcones(self, filename: str) -> Tuple[np.ndarray, np.array]:
        """
        Read the lightcone data from a file.

        Args:
            filename (str): The filename of the lightcone data.

        Returns:
            Tuple[np.ndarray, np.array]: The brightness temperature data and the parameters.
        """
        cone = np.load(filename)
        brightness_temp = cone['image']
        label = cone['label']
        lc_redshifts = cone['image_redshifts']
        gxH = cone['node_gxH']
        gxH_redshifts = cone['node_redshifts']
        return brightness_temp, label, lc_redshifts, gxH, gxH_redshifts

    def save(self, filename: str, brightness_temp: np.ndarray, label: np.array, lc_redshifts: np.array, gxH: np.array, gxH_redshifts: np.array) -> None:
        """
        Save the brightness temperature data and labels to a file.

        Args:
            filename (str): The filename to save the data.
            brightness_temp (np.ndarray): The brightness temperature data.
            label (np.array): The simulation parameters.
        """
        np.savez(filename, image=brightness_temp, label=label, image_redshifts=lc_redshifts, node_gxH=gxH, node_redshifts=gxH_redshifts)
        logging.info(f'Mock lightcone saved to {filename}')

    def convert_lightcones(self) -> None:
        """
        Convert existing lightcone simulations to add noise.
        """
        params = self.params['convert']
        source_path = params['source']
        destination_path = self.params['destination_noise']

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        files = glob.glob(os.path.join(source_path, '*.npz'))
        files.sort()  # Sort the files to ensure consistent processing order

        if params['n_sims'] == 'all':
            num_files = len(files)
        else:
            num_files = int(params['n_sims'])
            if num_files > len(files):
                logging.warning(f"Requested {num_files} simulations, but only {len(files)} are available. Processing all available files.")
                num_files = len(files)
            files = files[:num_files]

        logging.info(f'Converting {num_files} lightcones using multiprocessing')

        # Prepare arguments for the worker function
        args_list = [(self, file, destination_path) for file in files]

        # Create a pool of worker processes
        num_processes = multiprocessing.cpu_count()  # Or specify a number like 4
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process_file, args_list)

        logging.info('All tasks completed')

    def main(self) -> None:
        """
        Main method to create and/or convert lightcones based on the parameters.
        """
        if self.params['convert']['do_it']:
            self.convert_lightcones()
        logging.info('All tasks completed')

# Run the Noise class
if __name__ == '__main__':
    noise = Noise()
    noise.main()
