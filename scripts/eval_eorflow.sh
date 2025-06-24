#!/bin/bash
#SBATCH --job-name=eval_flow       # Job name
#SBATCH --time=1:00:00                             # Wall time
#SBATCH --partition=gshort                           # Partition (equivalent to PBS -q)
#SBATCH --gres=gpu:1                               # Request 1 GPU
#SBATCH --ntasks=1                                 # Number of tasks (equivalent to ppn)
#SBATCH --cpus-per-task=1                          # Number of CPUs per task
#SBATCH --mem-per-cpu=62G  # CPU memory (62G is fulfilled on all queues)
#SBATCH --output=eval_flow.o%A    # Standard output
#SBATCH --error=eval_flow.e%A     # Standard error

# Load CUDA module
module load cuda/11.4

# Activate the virtual environment
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

# Set CUDA device
mydev=`echo $CUDA_VISIBLE_DEVICES`
export CUDA_VISIBLE_DEVICES=$mydev

# Navigate to the training directory
cd /remote/gpu01a/pietschke/EoRFlow/src/evaluation

#nice -19 python inference_flow_only.py
nice -19 python eval_eorflow.py
echo "job done"
