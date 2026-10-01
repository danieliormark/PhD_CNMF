#!/bin/bash --login
#SBATCH --job-name=c13_crosscheck
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/logs/crosscheck_%A_%a.out
# Ticket 101: four tasks (usually different nodes) each run the same fixed fit
# with the portable-arithmetic settings used by submit_chunk13v9.sh. Compare
# the CROSSCHECK lines with each other and with the incline reference.
cd /mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE
python -u cross_machine_fit.py 2>/dev/null | grep CROSSCHECK
