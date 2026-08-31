import os

# Directory containing your noisy data files
directory = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10'

# Change to the directory
os.chdir(directory)

# List all files in the directory
for filename in os.listdir('.'):
    if filename.startswith('run_astro_') and filename.endswith('.npz'):
        new_name = filename.replace('run_astro_', '', 1)
        print(f"Renaming '{filename}' to '{new_name}'")
        os.rename(filename, new_name)
