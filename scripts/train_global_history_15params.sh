#!/bin/bash
#SBATCH --job-name=train_global_history_noise       # Job name
#SBATCH --time=3:00:00                             # Wall time
#SBATCH --partition=gshort                            # Partition (equivalent to PBS -q)
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --mem-per-cpu=62G  # CPU memory (62G is fulfilled on all queues)
#SBATCH --ntasks=1                                 # Number of tasks (equivalent to ppn)
#SBATCH --cpus-per-task=1                          # Number of CPUs per task
#SBATCH --output=train_global_history_noise.o%A    # Standard output
#SBATCH --error=train_global_history_noise.e%A     # Standard error

# Load CUDA module
module load cuda/11.4

# Activate the virtual environment
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

# Set CUDA device
mydev=`echo $CUDA_VISIBLE_DEVICES`
export CUDA_VISIBLE_DEVICES=$mydev

# Navigate to the training directory
cd /remote/gpu01a/pietschke/EoRFlow/src/training

# Run the training script
nice -19 python train_global_history_15params.py

echo "job done"
