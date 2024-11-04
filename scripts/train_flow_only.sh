#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=1:gpus=1:a30
#PBS -q a30
#PBS -N train_flow_only

module load cuda/11.4
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev

cd /remote/gpu01a/pietschke/EoRFlow/src/training

nice -19 python train_flow_only.py

echo "job done"
