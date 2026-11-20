#!/bin/bash --login
# Study B repeats — after all fit tasks end (ticket 107): alignment of the community numbering of every repeat stage
# (decided 2026-11-17) and the aggregation of B's pass 2 (as fitted and aligned; read by the λ rule). The other repeats
# are read by their analysis scripts on incline with the study option.
#SBATCH --job-name=m1_B_repeats_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_B_repeats_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_B_repeats_post_%j.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m fit.align --stages B_stage0_pass1 B_stage0_pass2 B_swap_bare B_start_test B_damping_test B_start_test2 --root "$OUT"
python -u -m fit.aggregate --root "$OUT" --stage B_stage0_pass2 --build "$BUILD"
M1_ALIGNED=1 python -u -m fit.aggregate --root "$OUT" --stage B_stage0_pass2 --build "$BUILD"
echo "study B repeats post-processing done"
