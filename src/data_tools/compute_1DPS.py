import numpy as np
import os
import py21cmfast as p21c
from py21cmfast_tools import calculate_ps
from joblib import Parallel, delayed

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

def process_file(file, gxH_redshifts, output_directory):
    data = np.load(file)
    image = data['image']
    tau = data['tau']
    gxH = data['gxH']

    xH_1 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=6.25, redshift_max=6.85)
    xH_2 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=6.85, redshift_max=7.56)
    xH_3 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=7.56, redshift_max=8.41)

    # Compute the redshifts
    redshifts = compute_redshifts(file)

    # Slices for which to compute the PS
    slice_1, redshifts_1 = select_slice(image, z1=6.25, z2=6.85, redshifts=redshifts)
    slice_2, redshifts_2 = select_slice(image, z1=6.85, z2=7.56, redshifts=redshifts)
    slice_3, redshifts_3 = select_slice(image, z1=7.56, z2=8.41, redshifts=redshifts)

    # Compute the power spectra
    PS1 = calculate_ps(slice_1, redshifts_1, box_length=200, box_side_shape=140, chunk_size=slice_1.shape[2]-1, calc_2d=False, calc_1d=True, calc_global=True, log_bins=True, nbins_1d=10)
    PS2 = calculate_ps(slice_2, redshifts_2, box_length=200, box_side_shape=140, chunk_size=slice_2.shape[2]-1, calc_2d=False, calc_1d=True, calc_global=True, log_bins=True, nbins_1d=10)
    PS3 = calculate_ps(slice_3, redshifts_3, box_length=200, box_side_shape=140, chunk_size=slice_3.shape[2]-1, calc_2d=False, calc_1d=True, calc_global=True, log_bins=True, nbins_1d=10)

    # Add a third dimension to the 1D power spectra
    power_1 = np.expand_dims(PS1['ps_1D'], axis=-1)
    power_1 = np.nan_to_num(power_1, nan=0.0)
    power_2 = np.expand_dims(PS2['ps_1D'], axis=-1)
    power_2 = np.nan_to_num(power_2, nan=0.0)
    power_3 = np.expand_dims(PS3['ps_1D'], axis=-1)
    power_3 = np.nan_to_num(power_3, nan=0.0)

    # Concatenate and save results
    power_spectra = np.concatenate([power_1, power_2, power_3], axis=0)

    # Extract the k bins
    kbins = np.array([PS1['k'], PS1['k'], PS3['k']])
    kbins = np.nan_to_num(kbins, nan=0.0)

    # Extract scalar values from global_Tb (if they are arrays)
    global_Tb_1 = PS1['global_Tb'][0] if isinstance(PS1['global_Tb'], (list, np.ndarray)) else PS1['global_Tb']
    global_Tb_2 = PS2['global_Tb'][0] if isinstance(PS2['global_Tb'], (list, np.ndarray)) else PS2['global_Tb']
    global_Tb_3 = PS3['global_Tb'][0] if isinstance(PS3['global_Tb'], (list, np.ndarray)) else PS3['global_Tb']

    # Now construct the label
    label = np.array([xH_1, xH_2, xH_3, redshifts_1[0], redshifts_2[0], redshifts_3[0], global_Tb_1, global_Tb_2, global_Tb_3])

    # Save the power spectra in a .npz file in the output directory
    output_file = os.path.join(output_directory, os.path.basename(file).replace('.npz', '.npz'))
    np.savez(output_file, image=power_spectra, label=label, tau=tau, gxH=gxH, kbins=kbins)
    print(f"Power spectra for {file}: {power_spectra.shape}")


# Directory containing the .npz files
directory = '/remote/gpu01a/heneka/21cmlightcones/pure_simulations'

# Directory to save the output .npz files
output_directory = '/remote/gpu01a/pietschke/SKA_flow/1D_data/train_2'
os.makedirs(output_directory, exist_ok=True)

# List all .npz files in the directory
npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
num_files = 10
# Select only the first num_files files [:num_files] - training
# Skip the first num_files files [num_files:] - testing
npz_files = npz_files[:num_files]

gxH_redshifts = np.load('./redshifts5.npy')

# Parallelize the loop
Parallel(n_jobs=4)(delayed(process_file)(file, gxH_redshifts, output_directory) for file in npz_files)









"""
# Directory containing the .npz files
directory = '/remote/gpu01a/heneka/21cmlightcones/pure_simulations'

# Directory to save the output .npz files
output_directory = '/remote/gpu01a/pietschke/SKA_flow/1D_data/test'
os.makedirs(output_directory, exist_ok=True)

# List all .npz files in the directory
npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]


num_files = 4000
# Select only the first num_files files [:num_files] - training
# Skip the first num_files files [num_files:] - testing
npz_files = npz_files[num_files:]

gxH_redshifts = np.load('./redshifts5.npy')

# Loop through each file, extract the field, compute redshifts, and calculate power spectra
for file in npz_files:
    # Load the .npz file
    data = np.load(file)
    
    # Extract the field
    image = data['image']
    tau = data['tau']
    gxH = data['gxH']

    xH_1 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=6.25, redshift_max=6.85)
    xH_2 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=6.85, redshift_max=7.56)
    xH_3 = average_xHI_in_range(gxH_redshifts, gxH, redshift_min=7.56, redshift_max=8.41)
    
    # Compute the redshifts
    redshifts = compute_redshifts(file)

    # Slices for which to compute the PS
    slice_1, redshifts_1 = select_slice(image, z1=6.25, z2=6.85, redshifts=redshifts)
    slice_2, redshifts_2 = select_slice(image, z1=6.85, z2=7.56, redshifts=redshifts) 
    slice_3, redshifts_3 = select_slice(image, z1=7.56, z2=8.41, redshifts=redshifts)

    # Compute the power spectra
    PS1 = calculate_ps(slice_1, redshifts_1, box_length=200, box_side_shape=140, chunk_size=slice_1.shape[2]-1, calc_2d=False, calc_1d=True, calc_global=True, log_bins=True, nbins_1d=10)
    PS2 = calculate_ps(slice_2, redshifts_2, box_length=200, box_side_shape=140, chunk_size=slice_2.shape[2]-1, calc_2d=False, calc_1d=True, calc_global=True, log_bins=True, nbins_1d=10)
    PS3 = calculate_ps(slice_3, redshifts_3, box_length=200, box_side_shape=140, chunk_size=slice_3.shape[2]-1, calc_2d=False, calc_1d=True, calc_global=True, log_bins=True, nbins_1d=10)

    # Add a third dimension to the 1D power spectra
    power_1 = np.expand_dims(PS1['ps_1D'], axis=-1)  # Adding a new axis at the end
    power_1 = np.nan_to_num(power_1, nan=0.0)
    
    power_2 = np.expand_dims(PS2['ps_1D'], axis=-1)  
    power_2 = np.nan_to_num(power_2, nan=0.0)
    
    power_3 = np.expand_dims(PS3['ps_1D'], axis=-1)  
    power_3 = np.nan_to_num(power_3, nan=0.0)

    # Concatenate and save results
    power_spectra = np.concatenate([power_1, power_2, power_3], axis=0)
    label = np.array([xH_1, xH_2, xH_3, PS1['redshifts'][0], PS2['redshifts'][0], PS3['redshifts'][0], PS1['global_Tb'][0], PS2['global_Tb'][0], PS3['global_Tb'][0]])
    
    # Extract the k bins
    kbins = np.array([PS1['k'], PS1['k'], PS3['k']])
    kbins = np.nan_to_num(kbins, nan=0.0)
    
    # Save the power spectra in a .npz file in the output directory
    output_file = os.path.join(output_directory, os.path.basename(file).replace('.npz', '.npz'))
    np.savez(output_file, image=power_spectra, label=label, tau=tau, gxH=gxH, kbins=kbins)
    # Do something with the power spectra (e.g., save or analyze)
    # For now, we'll just print the shape of the power spectra
    print(f"Power spectra for {file}: {power_spectra.shape}")

print('done')
"""