#!/bin/bash
#PBS -l walltime=500:00:00
#PBS -l nodes=1:ppn=8:bigmemlong
#PBS -q bigmemlong
#PBS -l mem=200gb,vmem=200gb
#PBS -N compute_ps

module load anaconda/3.0a
source /remote/gpu01a/pietschke/EoRFlow/venv_flow/bin/activate

cd /remote/gpu01a/pietschke/EoRFlow/src/data_tools/

nice -19 python compute_2DPS_xH_history.py


