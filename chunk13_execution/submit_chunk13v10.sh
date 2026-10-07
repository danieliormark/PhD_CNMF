#!/bin/bash --login
#SBATCH --job-name=chunk13_v10_t2
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/logs/chunk13v10_%A_%a.out
#SBATCH --error=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/logs/chunk13v10_%A_%a.err

# Ticket 105 (v10): slice T2, one array task per config and mode.
#   tasks 0-5  : C1..C6 with the temporal prior from the v9.4 T1 knee models
#                -> results/v10.0.t2_v2.prior/
#   tasks 6-11 : C1..C6 without a prior (free T2 run, the comparison baseline)
#                -> results/v10.0.t2_v2.free/
# After all tasks: python merge_chunk13_reports.py --results <each dir>, then
# python select_models.py --results <each dir>, then temporal_compare.py.
#
# LAMBDA_TEMPORAL must be set from the calibration (test T5, CLAUDE.md §4.25) before
# this is submitted; chunk13v10.py refuses a prior run without a positive weight.
LAMBDA_TEMPORAL=${LAMBDA_TEMPORAL:?set LAMBDA_TEMPORAL from the T5 calibration, e.g. sbatch --export=ALL,LAMBDA_TEMPORAL=0.1 submit_chunk13v10.sh}

CONFIGS=(C1 C2 C3 C4 C5 C6)
CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID % 6))]}
EXEC=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution
OUT=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs
if [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    MODE_ARGS="--prior-selection $EXEC/results/v9.4.t1_v2/model_selection.json --prior-data $OUT/Star_extended_matrices_t1_v2.pkl --lambda-temporal $LAMBDA_TEMPORAL"
    MODE=prior
else
    MODE_ARGS=""
    MODE=free
fi

cd "$EXEC"
mkdir -p logs results

source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:$LD_LIBRARY_PATH

# Single-threaded BLAS/OMP (ticket 76) and portable arithmetic (ticket 101), as v9.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=default
export MKL_CBWR=COMPATIBLE

echo "Task $SLURM_ARRAY_TASK_ID: config $CONFIG, mode $MODE, lambda_temporal ${LAMBDA_TEMPORAL} on $(hostname), $(date -Is)"
python -u chunk13v10.py --config "$CONFIG" --data "$OUT/Star_extended_matrices_t2_v2.pkl" $MODE_ARGS
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $? at $(date -Is)"
