#!/bin/bash --login
# Check W4 — after all fit tasks end: the paired difference-in-differences of the swap reading (fit/w4_swap_analysis.py).
#SBATCH --job-name=m1_w4_swap_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_w4_swap_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_w4_swap_post_%j.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m fit.w4_swap_analysis --root "$OUT" --out fit/results/stage0/w4_swap.json > fit/results/stage0/w4_swap.log
echo "W4 swap post-processing done: fit/results/stage0/w4_swap.{json,log}"
