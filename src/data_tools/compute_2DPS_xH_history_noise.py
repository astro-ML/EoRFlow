import numpy as np
import os
import py21cmfast as p21c
from py21cmfast_tools import calculate_ps
from multiprocessing import Pool, cpu_count

def compute_redshifts(image_file):
    cone = np.load(image_file)
    image = cone['image']
    label = cone['label']
    cosmo_params = p21c.CosmoParams(OMm=label[1])
    astro_params = p21c.AstroParams(INHOMO_RECO=True)
    user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
    flag_options = p21c.FlagOptions()
    sim_lightcone = p21c.LightCone(
        5.0,
        user_params,
        cosmo_params,
        astro_params,
        flag_options,
        0,
        {"brightness_temp": image},
        35.05
    )
    redshifts = sim_lightcone.lightcone_redshifts
    return redshifts

def get_xH_values(node_redshifts, global_xHI, target_redshifts):
    closest_xHI_values = []
    for target_redshift in target_redshifts:
        closest_index = np.abs(np.array(node_redshifts) - target_redshift).argmin()
        closest_xHI_values.append(global_xHI[closest_index])
    return closest_xHI_values

def check_and_recompute_ps(ps_result, slice_data, redshifts, box_length, box_side_shape, calc_params):
    ps_2d = ps_result['final_ps_2D']
    if ps_2d.shape == (1, 10, 9):
        print(f"Power spectrum has shape {ps_2d.shape}. Recomputing with k_par_bins=11.")
        new_ps_result = calculate_ps(
            slice_data,
            redshifts,
            box_length=box_length,
            box_side_shape=box_side_shape,
            chunk_size=slice_data.shape[2] - 1,
            calc_2d=True,
            calc_1d=False,
            calc_global=False,
            log_bins=True,
            nbins=10,
            kpar_bins=11
        )
        return new_ps_result['final_ps_2D']
    return ps_2d

def process_file(file_info):
    image_file, xh_file, output_directory, ps_redshifts = file_info

    # Load image data
    data_image = np.load(image_file)
    image = data_image['image']
    #tau = data_image['tau']

    # Load xH data
    data_xh = np.load(xh_file)
    #gxH = data_xh['gxH']  # For Benedikt's data
    gxH = data_xh['node_gxH']  # Uncomment if using Lara's data
    gxH_redshifts = data_xh['node_redshifts']  # Uncomment if using Lara's data
    #gxH_redshifts = np.load('./redshifts5.npy')  # For Benedikt's data

    # Compute lightcone redshifts
    #lc_redshifts = compute_redshifts(image_file)  # For Benedikt's data
    lc_redshifts = data_image['image_redshifts']  # Uncomment if using Lara's data

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
    output_file = os.path.join(output_directory, f'{os.path.basename(image_file).replace(".npz", "")}.npz')
    np.savez(output_file, image=power_spectra, label=label, redshifts=label_redshifts, gxH=gxH)
    print(f"Processed {image_file} with power spectra shape: {power_spectra.shape}")
    print(f"Processed {image_file} with redshifts: {label_redshifts}")

def main():
    # Directories containing the .npz files
    #image_directory = '/remote/gpu01a/heneka/21cmlightcones/opt_simulations'  # Benedikt's image data
    #xh_directory = '/remote/gpu01a/heneka/21cmlightcones/pure_simulations'    # Benedikt's xH data
    noise_data = '/remote/gpu01a/pietschke/lightcones/mock_astro' # Laras data
    image_directory =  noise_data
    xh_directory = noise_data
    output_directory = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_noise_astro'
    os.makedirs(output_directory, exist_ok=True)

    # List all .npz files in the image directory
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.npz')]
    num_files = 4000  # Adjust as needed

    # Select files for processing
    # For testing, skip the first num_files files
    # Select only the first num_files files [:num_files] - training
    # Skip the first num_files files [num_files:] - testing
    image_files = image_files[:num_files]

    # Define target redshift intervals
    ps_redshifts = np.arange(5, 20, 0.5)

    # Prepare arguments for processing
    file_info_list = []
    for image_file in image_files:
        filename = os.path.basename(image_file)
        xh_file = os.path.join(xh_directory, filename)
        if os.path.exists(xh_file):
            file_info_list.append((image_file, xh_file, output_directory, ps_redshifts))
        else:
            print(f"Warning: xH file {xh_file} does not exist. Skipping {image_file}.")

    # Process files sequentially
    for file_info in file_info_list:
        process_file(file_info)


    '''
    # Prepare arguments for parallel processing
    file_info_list = []
    for image_file in image_files:
        filename = os.path.basename(image_file)
        xh_file = os.path.join(xh_directory, filename)
        if os.path.exists(xh_file):
            file_info_list.append((image_file, xh_file, output_directory, ps_redshifts))
        else:
            print(f"Warning: xH file {xh_file} does not exist. Skipping {image_file}.")

    # Use multiprocessing to process files in parallel
    num_processors = min(cpu_count(), len(file_info_list))
    with Pool(num_processors) as pool:
        pool.map(process_file, file_info_list)
    '''


if __name__ == "__main__":
    main()
