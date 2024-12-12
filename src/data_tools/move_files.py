import os
import random
import shutil

# Define the source (training) and destination (testing) directories
src_dir = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10_2'
dst_dir = '/remote/gpu01a/pietschke/EoRFlow/data/2DPS_data/global_history/train_z5_20_10x10'

# Ensure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# List all files in the source directory
files = os.listdir(src_dir)

# Randomly select 700 files to move
files_to_move = random.sample(files, 4000)

# Move each selected file to the destination directory
for file_name in files_to_move:
    src_path = os.path.join(src_dir, file_name)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.move(src_path, dst_path)

print(f"Moved {len(files_to_move)} files from {src_dir} to {dst_dir}")
