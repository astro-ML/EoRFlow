#!/bin/bash
#SBATCH --job-name=eorflow_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err


echo "=================================================="
echo "EoRFlow Evaluation & Plotting"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
module load mpi/openmpi/4.1-gnu-14.2
module load devel/miniforge 
conda activate eor_env


# Set working directory
cd /pfs/10/work/hd_pt254-skatr/EoRFlow-dev

# Fixed evaluation config (same style as run_train.sh)
CONFIG_PATH="config/eval_skatr_loreli_trial.yaml"
export CONFIG_PATH

# Read config and derive output directory robustly.
# For sample.py, outputs are written to model_dir/samples.npz.
OUTPUT_DIR=$(python - <<'PY'
import os, yaml
c = yaml.safe_load(open(os.environ['CONFIG_PATH'])) or {}
if 'output_dir' in c and c['output_dir']:
    print(c['output_dir'])
else:
    mp = c.get('model_path', '')
    print(mp if os.path.isdir(mp) else os.path.dirname(mp))
PY
)
SAMPLES_FILE="${OUTPUT_DIR}/samples.npz"
N_PANELS=$(python - <<'PY'
import os, yaml
c = yaml.safe_load(open(os.environ['CONFIG_PATH'])) or {}
print(c.get('plot_n_panels', 5))
PY
)

echo ""
echo "Running evaluation..."
echo "Config: ${CONFIG_PATH}"
echo ""

python src/evaluation/sample.py --config "${CONFIG_PATH}"

if [ ! -f "${SAMPLES_FILE}" ]; then
    echo "ERROR: samples file not found at ${SAMPLES_FILE}"
    exit 1
fi

echo ""
echo "Creating diagnostic plots..."
echo "Samples: ${SAMPLES_FILE}"
echo ""

python src/evaluation/plotting.py \
    --samples "${SAMPLES_FILE}" \
    --n_panels ${N_PANELS}

echo ""
echo "=================================================="
echo "Evaluation & Plotting Complete"
echo "Output directory: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "=================================================="
