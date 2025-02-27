#!/bin/bash
#SBATCH --job-name=train_flow       # Job name
#SBATCH --time=6:00:00                             # Wall time
#SBATCH --partition=a30                            # Partition (equivalent to PBS -q)
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --ntasks=1                                 # Number of tasks (equivalent to ppn)
#SBATCH --cpus-per-task=1                          # Number of CPUs per task
#SBATCH --output=train_flow.o%A    # Standard output
#SBATCH --error=train_flow.e%A     # Standard error

# Load CUDA module
module load cuda/11.4

# Activate the virtual environment
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

# Set CUDA device
mydev=`echo $CUDA_VISIBLE_DEVICES`
export CUDA_VISIBLE_DEVICES=$mydev

# Navigate to the training directory
cd /remote/gpu01a/pietschke/EoRFlow/src/training

nice -19 python train_flow_only.py

echo "job done"
