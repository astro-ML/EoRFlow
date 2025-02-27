#!/bin/bash
#PBS -l walltime=200:00:00
#PBS -l nodes=1:ppn=8
#PBS -q bigmemlong
#PBS -l mem=120gb,vmem=120gb
#PBS -N create_mocks

module load anaconda/3.0a
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

cd /remote/gpu01a/pietschke/EoRFlow/src/data_tools/

nice -19 python add_noise_h5.py


