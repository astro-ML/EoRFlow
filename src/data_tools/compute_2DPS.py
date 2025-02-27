import numpy as np
import os
import py21cmfast as p21c
from py21cmfast_tools import calculate_ps

def compute_redshifts(file):
    """
    Compute redshift from .npz file with keys: image, label, tau, gxH.
    Note: assumes specific boxlength, resolution and redshift scope of the simulation!
    :param file: Name of the .npz file to read in.
    """
    cone = np.load(file)
    image = cone['image']
    label = cone['label']
    cosmo_params = p21c.CosmoParams(OMm=label[1])
    astro_params = p21c.AstroParams(INHOMO_RECO=True)
    user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
    flag_options = p21c.FlagOptions()
    sim_lightcone = p21c.LightCone(5., user_params, cosmo_params, astro_params, flag_options, 0, {"brightness_temp": image}, 35.05)
    redshifts = sim_lightcone.lightcone_redshifts

    return redshifts

def select_slice(image, z1, z2, redshifts):
    """
    Select redshift slice from brightness temperature image.
    : param image:   Image to select slice from.
    : z1, z2 : define the redshift interval.
    : redshifts : Lightcone redshifts associated to the image.
    """
    # Get indices that correspond to redshift range
    start = np.searchsorted(redshifts, z1, side='left') 
    end = (np.searchsorted(redshifts, z2, side='right') - 1 ) 
    return image[:, :, start:end], redshifts[start:end]

def average_xHI_in_range(node_redshifts, global_xHI, redshift_min, redshift_max):
    indices_in_range = [i for i, z in enumerate(node_redshifts) if redshift_min <= z <= redshift_max]
    if not indices_in_range:
        return None
    xHI_in_range = [global_xHI[i] for i in indices_in_range]
    return sum(xHI_in_range) / len(xHI_in_range)

def check_and_recompute_ps(ps_result, slice_data, redshifts, box_length, box_side_shape, calc_params):
    """
    Check if the computed power spectrum has the shape (1, 10, 9) and recompute with k_par_bins=11.
    """
    ps_2d = ps_result['final_ps_2D']
    if ps_2d.shape == (1, 10, 9):
        print(f"Power spectrum has shape {ps_2d.shape}. Recomputing with k_par_bins=11.")
        # Recompute with k_par_bins=11
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
            kpar_bins=11  # Recompute with one extra k_par bin
        )
        return new_ps_result['final_ps_2D']
    return ps_2d

# Directory containing the .npz files
#directory = '/remote/gpu01a/heneka/21cmlightcones/pure_simulations' # Benedikt
#directory = '/remote/gpu01a/heneka/21cmlightcones/pure_simulations_astro' # Lara
directory = '/remote/gpu01a/pietschke/lightcones/mock_astro' # Lara

# Directory to save the output .npz files
output_directory = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_noise'
os.makedirs(output_directory, exist_ok=True)

# List all .npz files in the directory
npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]

#num_files = 10
# Select only the first num_files files [:num_files] - training
# Skip the first num_files files [num_files:] - testing
#npz_files = npz_files[:num_files]

# read in node redshifts for gxH
#gxH_redshifts = np.load('./redshifts5.npy') #Benedikt
file_counter = 1

# Loop through each file, extract the field, compute redshifts, and calculate power spectra
for file in npz_files:
    # Load the .npz file
    data = np.load(file)
    image = data['image']
    #tau = data['tau']
    #gxH = data['gxH'] # Benedikt
    gxH = data['node_gxH'] # Lara
    gxH_redshifts = data['node_redshifts'] # Lara

    xH_1 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=6.25, redshift_max=6.85)
    xH_2 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=6.85, redshift_max=7.56)
    xH_3 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=7.56, redshift_max=8.41)
    
    # Compute the redshifts
    #redshifts = compute_redshifts(file) #Benedikt
    redshifts = data['image_redshifts'] #Lara
    
    # Slices for which to compute the PS
    slice_1, redshifts_1 = select_slice(image, z1=6.25, z2=6.85, redshifts=redshifts)
    slice_2, redshifts_2 = select_slice(image, z1=6.85, z2=7.56, redshifts=redshifts) 
    slice_3, redshifts_3 = select_slice(image, z1=7.56, z2=8.41, redshifts=redshifts)
    
    # Compute the power spectra
    PS1 = calculate_ps(slice_1, redshifts_1, box_length=200, box_side_shape=140, chunk_size=slice_1.shape[2]-1, calc_2d=True, calc_1d=False, calc_global=False, log_bins=True, nbins=10, kpar_bins=10)
    PS2 = calculate_ps(slice_2, redshifts_2, box_length=200, box_side_shape=140, chunk_size=slice_2.shape[2]-1, calc_2d=True, calc_1d=False, calc_global=False, log_bins=True, nbins=10, kpar_bins=10)
    PS3 = calculate_ps(slice_3, redshifts_3, box_length=200, box_side_shape=140, chunk_size=slice_3.shape[2]-1, calc_2d=True, calc_1d=False, calc_global=False, log_bins=True, nbins=10, kpar_bins=10)
    
    # Check and recompute the power spectrum if needed
    PS1_2D = check_and_recompute_ps(PS1, slice_1, redshifts_1, box_length=200, box_side_shape=140, calc_params=None)
    PS2_2D = check_and_recompute_ps(PS2, slice_2, redshifts_2, box_length=200, box_side_shape=140, calc_params=None)
    PS3_2D = check_and_recompute_ps(PS3, slice_3, redshifts_3, box_length=200, box_side_shape=140, calc_params=None)
    
    # Concatenate and save results
    power_spectra = np.concatenate([PS1_2D, PS2_2D, PS3_2D], axis=0)
    label = np.array([xH_1, xH_2, xH_3, PS1['redshifts'][0], PS2['redshifts'][0], PS3['redshifts'][0]])

    # Save the power spectra in a .npz file in the output directory
    #output_file = os.path.join(output_directory, os.path.basename(file).replace('.npz', '.npz'))
    output_file = os.path.join(output_directory, f'run_astro_{file_counter}.npz')
    np.savez(output_file, image=power_spectra, label=label, gxH=gxH)

    print(f"Power spectra for {file}: {power_spectra.shape}")

    file_counter += 1

print('done')
