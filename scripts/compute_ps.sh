#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=4
#PBS -q medium_bookworm
#PBS -l mem=8gb,vmem=8gb
#PBS -N compute_ps

module load anaconda/3.0a
source /remote/gpu01a/pietschke/SKA_flow/venv_flow/bin/activate

cd /remote/gpu01a/pietschke/SKA_flow

nice -19 python compute_2DPS_xH_history.py


