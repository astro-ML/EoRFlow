import numpy as np
import os
import py21cmfast as p21c
from py21cmfast_tools import calculate_ps
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
log_filename = 'compute_ps.log'
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_xH_values(node_redshifts, global_xHI, target_redshifts):
    closest_xHI_values = []
    for target_redshift in target_redshifts:
        closest_index = np.abs(np.array(node_redshifts) - target_redshift).argmin()
        closest_xHI_values.append(global_xHI[closest_index])
    return closest_xHI_values

def process_file(file_info):
    file, output_directory, ps_redshifts = file_info
    output_file = os.path.join(output_directory, f'{os.path.basename(file).replace(".npz", "")}.npz')
    
    # Skip if output already exists
    if os.path.exists(output_file):
        logging.info(f"Skipping {file}, output already exists.")
        return
    
    try:
        data = np.load(file)
        image = data['image']
        gxH = data['gxH']
        gxH_redshifts = data['gxH_redshifts']
        lc_redshifts = data['lc_redshifts']
        params = data['label']
        
        # Compute xH values for specific redshifts
        xH_values = get_xH_values(gxH_redshifts, gxH, ps_redshifts)
        
        # Calculate power spectrum
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
        
        power_spectra = ps_result['final_ps_2D']
        label_redshifts = ps_result['redshifts']
        label = np.array(xH_values)
        
        # Save the power spectra and labels
        np.savez(output_file, image=power_spectra, label=label, redshifts=label_redshifts, gxH=gxH, params=params)
        logging.info(f"Processed {file} successfully.")
    
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")

def main():
    directory = '/remote/gpu01a/pietschke/lightcones/mock_tom'
    output_directory = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/toms_data_noise'
    os.makedirs(output_directory, exist_ok=True)

    # List all .npz files in the directory
    npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
    
    # Define target redshift intervals
    ps_redshifts = np.arange(5, 20, 0.5)
    
    # Set the number of files to process
    num_files_to_process = 'all'  # Change this to an integer or 'all'
    
    if num_files_to_process != 'all':
        npz_files = npz_files[:int(num_files_to_process)]
    
    # Prepare arguments for parallel processing
    file_info_list = [(file, output_directory, ps_redshifts) for file in npz_files]
    
    # Use multiprocessing to process files in parallel
    num_processors = min(cpu_count(), len(npz_files))
    with Pool(num_processors) as pool:
        pool.map(process_file, file_info_list)

if __name__ == "__main__":
    main()


  
    
   
