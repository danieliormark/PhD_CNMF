#!/bin/bash --login
# Stage 1 — after all fit tasks end: alignment of the community numbering and the aggregation (as fitted and aligned).
# Env: OUT (fit root), STAGE.
#SBATCH --job-name=m1_stage1_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_stage1_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_stage1_post_%j.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m fit.align --stages "$STAGE" --root "$OUT"
python -u -m fit.aggregate --root "$OUT" --stage "$STAGE"
M1_ALIGNED=1 python -u -m fit.aggregate --root "$OUT" --stage "$STAGE"
echo "stage-1 post-processing done: $OUT/$STAGE/_aggregate*.json"
