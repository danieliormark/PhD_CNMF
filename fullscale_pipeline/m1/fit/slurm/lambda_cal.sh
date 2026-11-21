#!/bin/bash
# λ calibration, study A (ticket 107; owner 2026-11-21, point 1, RL-330): new r at every coarse K, both approaches —
# 0.45 / 0.6 / 0.8 where pass 3 chose r 0.3 (K 3, 5; K 8 approach 1), 1.5 / 2 / 3 where it chose 1 (K 8 approach 2, K 13, 20);
# 5 groups × 10 seeds; settings as pass 3 (KL, S3 10 %, 8,000-epoch ceiling); 1,500 fits, ≈ 1,100–1,300 CPU-hours.
# 188 array tasks × 8 fits, 12 h per task. Then alignment, aggregation, readings and the λ rule R2 over pass 3's cells and
# the new ones → fit/results/stage0/lambda_cal_choice_A.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/lambda_cal.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_lambda_cal.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=lambda_cal PASS3=stage0_pass3 STUDY=A
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=24:00:00 --mem-per-cpu=8G --export=ALL submit_lambda_cal_post.sh)
echo "λ-calibration post-processing job $PID (starts after all fit tasks end)"
