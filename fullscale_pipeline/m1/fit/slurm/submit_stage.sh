#!/bin/bash --login
# M1 fits as a SLURM array on CSF3 (ticket 107, F8). One array task runs PER_TASK fits in parallel, one thread each.
# Submit (from a CSF3 login node, after the owner's permission):
#   JOBS=/path/jobs.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 \
#   sbatch --array=0-$(( ( $(wc -l < $JOBS) + 8 - 1 ) / 8 - 1 )) --cpus-per-task=8 --mem-per-cpu=8G --time=48:00:00 submit_stage.sh
# Partition multicore (AMD Genoa, 8 GB per core, 2–168 cores, 7 days; CSF3 docs, checked 2026-11-09).
#SBATCH --job-name=m1_fits
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_%A_%a.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_%A_%a.err

set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
# one thread per fit and reproducible arithmetic across machines (tickets 76, 101)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE
python -u -m fit.run_fit --jobs "$JOBS" --out "$OUT" --task "$SLURM_ARRAY_TASK_ID" --per-task "${PER_TASK:-8}"
