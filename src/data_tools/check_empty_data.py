import os
import numpy as np

# Define the directory containing the .npz files
directory = "/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_noise"

# Get a list of .npz files in the directory
npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]

# Counters
zero_image_count = 0
zero_label_count = 0
one_label_count = 0
total_files = len(npz_files)

# Iterate over the files and check the 'image' and 'label' keys
for file in npz_files:
    file_path = os.path.join(directory, file)
    try:
        data = np.load(file_path)

        # Check 'image' key for all zeros
        if 'image' in data:
            image = data['image']
            if np.all(image == 0):
                zero_image_count += 1

        # Check 'label' key for zeros and ones
        if 'label' in data:
            label = data['label']
            if np.any(label == 0):
                zero_label_count += 1
            if np.any(label == 1):
                one_label_count += 1

    except Exception as e:
        print(f"Error loading {file}: {e}")

# Print results
print(f"Total files checked: {total_files}")
print(f"Files where 'image' is entirely zero: {zero_image_count}")
print(f"Files where 'label' contains at least one 0: {zero_label_count}")
print(f"Files where 'label' contains at least one 1: {one_label_count}")