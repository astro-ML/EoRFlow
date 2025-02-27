import os
import numpy as np

folder = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'  # Replace with the actual folder path

# List all .npz files in the folder
npz_files = [f for f in os.listdir(folder) if f.endswith('.npz')]

for fname in npz_files:
    filepath = os.path.join(folder, fname)
    data = np.load(filepath)
    all_keys_ok = True

    arr = data['params']
    # Check if arr is exactly shape (30,)
    if arr.shape != (6,):
        print(f"Warning: In file {fname}, key 'params' has shape {arr.shape}, not (30,).")
        all_keys_ok = False
    data.close()
    #if all_keys_ok:
    #    print(f"All keys in {fname} have shape (30,).")
