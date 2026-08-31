#!/bin/bash
#SBATCH --job-name=eval_eorflow # name of job
#SBATCH --output=eval_eorflow%j.out # output file (%j = job ID)
#SBATCH --error=eval_eorflow%j.err # error file (%j = job ID)
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 2 nodes
#SBATCH --ntasks=1 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:1 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=1 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=01:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=ybg@a100 # A100 accounting
module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load arch/a100 # select modules compiled for A100
module load miniforge/24.9.0
module load pytorch-gpu/py3/2.3.0 # load modules

set -x # activate echo of launched commands


cd /lustre/fswork/projects/rech/ybg/uuv28wh/EoRFlow/src/evaluation
srun /linkhome/rech/genpic01/uuv28wh/.conda/envs/ska_env/bin/python inference_flow_only.py

echo 'job done'
