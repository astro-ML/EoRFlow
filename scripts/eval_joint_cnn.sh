#!/bin/bash
#SBATCH --job-name=eval_joint       # Job name
#SBATCH --time=1:00:00                             # Wall time
#SBATCH --partition=a30                           # Partition (equivalent to PBS -q)
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --ntasks=1                                 # Number of tasks (equivalent to ppn)
#SBATCH --cpus-per-task=1                          # Number of CPUs per task
#SBATCH --output=eval_joint.o%A    # Standard output
#SBATCH --error=eval_joint.e%A     # Standard error

# Load CUDA module
module load cuda/11.4

# Activate the virtual environment
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

# Set CUDA device
mydev=`echo $CUDA_VISIBLE_DEVICES`
export CUDA_VISIBLE_DEVICES=$mydev

# Navigate to the training directory
cd /remote/gpu01a/pietschke/EoRFlow/src/evaluation

nice -19 python inference_joint_cnn.py

echo "job done"
