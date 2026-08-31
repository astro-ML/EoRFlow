#!/bin/bash
#SBATCH --job-name=eorflow_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "=================================================="
echo "EoRFlow Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Activate conda environment
module load mpi/openmpi/4.1-gnu-14.2
module load devel/miniforge 
conda activate eor_env

# Set working directory
cd /pfs/10/work/hd_pt254-skatr/EoRFlow-dev

echo "Starting training..."

# Run training
python src/training/train.py \
    --config config/train_skatr_loreli_trial.yaml

echo ""
echo "=================================================="
echo "Training Complete"
echo "End time: $(date)"
echo "=================================================="
