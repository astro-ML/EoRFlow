import os
import numpy as np
import shutil

# Define directories
input_directory = "/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/test_10x10"
zero_flagged_directory = "/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/train_10x10_empty"

# Ensure the zero-flagged directory exists
os.makedirs(zero_flagged_directory, exist_ok=True)

# Get list of .npz files
npz_files = [f for f in os.listdir(input_directory) if f.endswith('.npz')]

# Counter for zero-image files
zero_image_count = 0
total_files = len(npz_files)

# Iterate over files and check 'image' key
for file in npz_files:
    file_path = os.path.join(input_directory, file)
    try:
        data = np.load(file_path)
        if 'image' in data:
            image = data['image']
            if np.all(image == 0):
                zero_image_count += 1
                # Move the file to the flagged directory
                shutil.move(file_path, os.path.join(zero_flagged_directory, file))
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Print results
print(f"Total files checked: {total_files}")
print(f"Files moved to flagged directory: {zero_image_count}")