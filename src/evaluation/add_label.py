import os
import numpy as np

pure_dir = '/remote/gpu01a/heneka/21cmlightcones/pure_simulations_astro'
noise_dir = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/test_z5_20_10x10_noise_astro'

# List all .npz files in the pure directory
pure_files = [f for f in os.listdir(pure_dir) if f.endswith('.npz')]

for fname in pure_files:
    pure_path = os.path.join(pure_dir, fname)
    noise_path = os.path.join(noise_dir, fname)

    # Check if corresponding file exists in the noise directory
    if not os.path.exists(noise_path):
        print(f"Warning: {fname} does not exist in the noise directory. Skipping.")
        continue

    # Load the label from the pure file
    try:
        pure_data = np.load(pure_path)
        if 'label' not in pure_data:
            print(f"Warning: 'label' not found in {pure_path}. Skipping.")
            pure_data.close()
            continue

        label = pure_data['label']
        pure_data.close()
    except Exception as e:
        print(f"Error loading {pure_path}: {e}")
        continue


    # Insert the value 2 at position 0
    label = np.insert(label, 0, 2)


    # Check that label has 6 elements
    if label.shape != (6,):
        print(f"Warning: 'label' in {pure_path} does not have shape (6,). It has shape {label.shape}. Skipping.")
        continue

    # Load the noise file and extract its contents
    try:
        noise_data = np.load(noise_path)
        # Convert to dict to re-save easily with new keys
        noise_dict = {key: noise_data[key] for key in noise_data.keys()}
        noise_data.close()
    except Exception as e:
        print(f"Error loading {noise_path}: {e}")
        continue

    # Add modified label as 'params'
    noise_dict['params'] = label

    # Re-save the noise file with params included
    try:
        np.savez(noise_path, **noise_dict)
        print(f"Added 'params' with shape {label.shape} to {noise_path}")
    except Exception as e:
        print(f"Error saving {noise_path}: {e}")
