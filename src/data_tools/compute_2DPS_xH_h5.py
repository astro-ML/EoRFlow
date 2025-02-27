import os
import numpy as np
import h5py
import py21cmfast as p21c
from py21cmfast_tools import calculate_ps  
import multiprocessing

# Paths
input_directory = '/remote/gpu01a/schlenker/21cm-wrapper/data_sbi'
output_directory = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/toms_data_pure'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define function to extract xH values
def get_xH_values(node_redshifts, global_xHI, target_redshifts):
    closest_xHI_values = []
    for target_redshift in target_redshifts:
        closest_index = np.abs(np.array(node_redshifts) - target_redshift).argmin()
        closest_xHI_values.append(global_xHI[closest_index])
    return np.array(closest_xHI_values)

# Define target redshifts for PS calculations
ps_redshifts = np.arange(5, 20, 0.5)

# Get list of .h5 files
h5_files = [f for f in os.listdir(input_directory) if f.endswith('.h5')]

# Set number of simulations to process
num_simulations = min(len(h5_files), 7000)  # Change 10 to desired number

# Define processing function
def process_file(h5_file):
    file_path = os.path.join(input_directory, h5_file)
    try:
        # Read lightcone
        lightcone = p21c.outputs.LightCone.read(file_path)
        
        # Extract parameters
        OM = lightcone.cosmo_params.OMm
        L_X = lightcone.astro_params.L_X
        T_vir = lightcone.astro_params.ION_Tvir_MIN
        E0 = lightcone.astro_params.NU_X_THRESH
        Zeta = lightcone.astro_params.HII_EFF_FACTOR
        mDM = 2  # not varied here

        params = np.array([mDM, OM, E0, L_X, T_vir, Zeta])
        
        # Extract relevant quantities
        lc_redshifts = lightcone.lightcone_redshifts
        gxH = np.flip(lightcone.global_xH)
        gxH_redshifts = np.flip(lightcone.node_redshifts)
        image = lightcone.brightness_temp
        
        # Compute xH values
        xH_values = get_xH_values(gxH_redshifts, gxH, ps_redshifts)
        
        # Compute power spectra
        ps_result = calculate_ps(
            lc=image,
            lc_redshifts=lc_redshifts,
            zs=ps_redshifts,
            box_length=200,
            box_side_shape=140,
            calc_2d=True,
            calc_1d=False,
            calc_global=False,
            log_bins=True,
            nbins=10,
            kpar_bins=10
        )
        
        PS_2D = ps_result['final_ps_2D']
        label_redshifts = ps_result['redshifts']
        label = np.array(xH_values)
        
        # Save as .npz
        output_file = os.path.join(output_directory, f'{h5_file.replace(".h5", ".npz")}')
        np.savez(output_file, image=PS_2D, label=label, redshifts=label_redshifts, params=params)
        print(f"Processed and saved: {output_file}")
    
    except Exception as e:
        print(f"Error processing {h5_file}: {e}")

# Set number of worker processes
num_workers = min(multiprocessing.cpu_count(), num_simulations)

# Run in parallel
if __name__ == "__main__":
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_file, h5_files[:num_simulations])
