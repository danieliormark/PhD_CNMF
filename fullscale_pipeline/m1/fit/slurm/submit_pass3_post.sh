#!/bin/bash --login
# Stage 0 pass 3 — after all fit tasks end (ticket 107, owner 2026-11-19): alignment of the community numbering
# (decided 2026-11-17), the aggregation as fitted and aligned, the J1 choice (RL-299) and the readings reported beside
# it. Env: OUT (fit root), STAGE, STUDY (A / B), BUILD (study B only).
#SBATCH --job-name=m1_pass3_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_pass3_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_pass3_post_%j.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
R=fit/results/stage0
python -u -m fit.align --stages "$STAGE" --root "$OUT"
python -u -m fit.aggregate --root "$OUT" --stage "$STAGE" ${BUILD:+--build "$BUILD"}
M1_ALIGNED=1 python -u -m fit.aggregate --root "$OUT" --stage "$STAGE" ${BUILD:+--build "$BUILD"}
python -u -m fit.pass3_readings --root "$OUT" --stage "$STAGE" --study "$STUDY" ${BUILD:+--build "$BUILD"} --workers 8 \
  --out $R/pass3_readings_$STUDY.json > $R/pass3_readings_$STUDY.log
python -u -m fit.pass3_choice --agg "$OUT/$STAGE/_aggregate_aligned.json" --readings $R/pass3_readings_$STUDY.json \
  --out $R/pass3_choice_$STUDY.json > $R/pass3_choice_$STUDY.log
echo "pass 3 post-processing done: $R/pass3_choice_$STUDY.{json,log}, $R/pass3_readings_$STUDY.{json,log}"
