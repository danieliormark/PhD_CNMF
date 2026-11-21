#!/bin/bash --login
# λ calibration — after all fit tasks end (ticket 107, owner 2026-11-21): alignment of the new fits, aggregation (as fitted
# and aligned), readings, and the λ rule R2 / K protocol over pass 3's cells (r 0 / 0.3 / 1) together with the new r.
# Env: OUT, STAGE (lambda_cal / B_lambda_cal), PASS3 (stage0_pass3 / B_stage0_pass3), STUDY (A / B), BUILD (B only).
#SBATCH --job-name=m1_lambda_cal_post
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_lambda_cal_post_%j.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_lambda_cal_post_%j.err
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
  --out $R/lambda_cal_readings_$STUDY.json > $R/lambda_cal_readings_$STUDY.log
python -u -m fit.pass3_choice --agg "$OUT/$PASS3/_aggregate_aligned.json" "$OUT/$STAGE/_aggregate_aligned.json" \
  --readings $R/pass3_readings_$STUDY.json $R/lambda_cal_readings_$STUDY.json \
  --out $R/lambda_cal_choice_$STUDY.json > $R/lambda_cal_choice_$STUDY.log
echo "λ calibration post-processing done: $R/lambda_cal_choice_$STUDY.{json,log}"
