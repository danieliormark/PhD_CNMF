#!/bin/bash --login
# child×child focused test (RL-378) — after all fit tasks end: alignment, aggregation (aligned), analysis.
#SBATCH --job-name=m1_cc_test_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_cc_test_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_cc_test_post_%j.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export M1_ENTITY_RULE=coverage
python -u -m fit.align --stages cc_test --root "$OUT"
M1_ALIGNED=1 python -u -m fit.aggregate --root "$OUT" --stage cc_test
python -u -m fit.cc_test_analysis --root "$OUT" --out fit/results/stage0/cc_test.json --workers 8 > fit/results/stage0/cc_test.log
echo "child×child test post-processing done: fit/results/stage0/cc_test.{json,log}"
