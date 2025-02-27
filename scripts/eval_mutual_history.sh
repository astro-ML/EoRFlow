#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -q gshort
#PBS -N eval_global_history

module load cuda/11.4
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev

cd /remote/gpu01a/pietschke/EoRFlow/src/evaluation

nice -19 python eval_mutual_files.py

echo "job done"
