#!/bin/bash --login
#SBATCH --job-name=chunk13_v9_grid
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/logs/chunk13v9_%A_%a.out
#SBATCH --error=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/logs/chunk13v9_%A_%a.err

# One array task per config (C1..C6). Each task runs the whole Module 4 chain
# for its config: scout over K_LIST, the hypervolume filter that picks the
# K values for the deep dive, archiving, and the stability phase. Model
# selection never compares across configs, so the tasks are independent.
# Splitting by config/K instead would break the hypervolume filter, which
# needs every K of one config. Each task writes its own report files
# (*_C<n>.json); combine them afterwards with
#   python merge_chunk13_reports.py
# --cpus-per-task=2 is the multicore partition's minimum; the code runs
# single-threaded (ticket 76), the second core is unused headroom.

CONFIGS=(C1 C2 C3 C4 C5 C6)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

cd /mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution
mkdir -p logs results

source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:$LD_LIBRARY_PATH

# Single-threaded BLAS/OMP (ticket 76): must match set_seeds()'s
# torch.set_num_threads(1), otherwise reduction order makes fits differ run to run.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Task $SLURM_ARRAY_TASK_ID: config $CONFIG on $(hostname), $(date -Is)"
python chunk13v9.py --config "$CONFIG"
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $? at $(date -Is)"
