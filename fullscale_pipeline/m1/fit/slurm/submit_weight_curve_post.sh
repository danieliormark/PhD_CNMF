#!/bin/bash --login
# Intermediate weighting check (RL-353) — after all fit tasks end: alignment, aggregation (aligned), the curve (weight_curve_analysis).
#SBATCH --job-name=m1_weight_curve_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_weight_curve_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_weight_curve_post_%j.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m fit.align --stages weight_curve --root "$OUT"
M1_ALIGNED=1 python -u -m fit.aggregate --root "$OUT" --stage weight_curve
python -u -m fit.weight_curve_analysis --root "$OUT" --out fit/results/stage0/weight_curve.json > fit/results/stage0/weight_curve.log
echo "weight-curve post-processing done: fit/results/stage0/weight_curve.{json,log}"
