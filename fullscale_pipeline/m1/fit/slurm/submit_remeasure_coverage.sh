#!/bin/bash --login
# Worker of remeasure_coverage.sh (item (l), coverage rule c = 0.5, RL-345). One array task per block of readings.
#SBATCH --job-name=m1_remeasure_cov
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_remeasure_cov_%A_%a.out
#SBATCH --error=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/logs/m1_remeasure_cov_%A_%a.err
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export M1_ENTITY_RULE=coverage
OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits
BUILD_B=/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/
R=fit/results/stage0
keep() { for f in "$@"; do [ -f "$f" ] || continue; b="${f%.*}_placed.${f##*.}"; [ -f "$b" ] || cp -p "$f" "$b"; done; }
case "$SLURM_ARRAY_TASK_ID" in
0|1)
  if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then S=A; STAGE=stage0_pass3; BA=""; else S=B; STAGE=B_stage0_pass3; BA="--build $BUILD_B"; fi
  keep "$OUT/$STAGE/_aggregate_aligned.json" "$R/pass3_choice_$S.json" "$R/pass3_choice_$S.log"
  M1_ALIGNED=1 python -u -m fit.aggregate --root "$OUT" --stage "$STAGE" $BA --workers 8
  python -u -m fit.pass3_choice --agg "$OUT/$STAGE/_aggregate_aligned.json" --readings "$R/pass3_readings_$S.json" \
    --out "$R/pass3_choice_$S.json" > "$R/pass3_choice_$S.log"
  ;;
2)
  keep $R/start_test.json $R/start_test.log $R/start_test2.json $R/start_test2.log \
       $R/start_test_supplementary.json $R/start_test_supplementary.log \
       $R/B/start_test.json $R/B/start_test.log $R/B/start_test2.json $R/B/start_test2.log
  python -u -m fit.start_test_analysis --out $R/start_test.json > $R/start_test.log
  python -u -m fit.start_test_analysis --include-unconverged --out $R/start_test_supplementary.json > $R/start_test_supplementary.log
  python -u -m fit.start_test2_analysis --out $R/start_test2.json > $R/start_test2.log
  M1_STUDY=B python -u -m fit.start_test_analysis --out $R/B/start_test.json > $R/B/start_test.log
  M1_STUDY=B python -u -m fit.start_test2_analysis --out $R/B/start_test2.json > $R/B/start_test2.log
  ;;
3)
  keep $R/stability_genre_tests.json $R/stability_genre_tests.log
  python -u -m fit.stability_genre_tests --out $R/stability_genre_tests.json --workers 8 > $R/stability_genre_tests.log
  ;;
4)
  for a in 1 2; do for b in 1 2; do
    keep $R/s3_gate1_a${a}_b${b}.json $R/s3_gate1_a${a}_b${b}.log
    python -u -m fit.s3_gate1 --approach $a --beta $b --out $R/s3_gate1_a${a}_b${b}.json > $R/s3_gate1_a${a}_b${b}.log &
  done; done; wait
  ;;
esac
echo "remeasurement task $SLURM_ARRAY_TASK_ID done"
