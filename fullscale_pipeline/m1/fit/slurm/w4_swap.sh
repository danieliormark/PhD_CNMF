#!/bin/bash
# Swap with the word-similarity term on — check W4 (ticket 107; owner 2026-11-26: "I approve the 70 fits and the paired
# criterion"; RL-405): term off (r 0) at p 1 for (approach, K) = (1, 5), (2, 5), (1, 13), (2, 13), (1, 20) and term on at
# p 1 for approach 1 K 13 r 0.3 and K 20 r 1.0; study A, W1, settings as pass 3, hold-out group 1, seeds 1–10; 70 fits
# ≈ 55 CPU-hours; 9 tasks × 8 fits, 12 h. Then the paired difference-in-differences (fit/w4_swap_analysis.py) →
# fit/results/stage0/w4_swap.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/w4_swap.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_w4_swap.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --cpus-per-task=2 --time=04:00:00 --mem-per-cpu=8G --export=ALL submit_w4_swap_post.sh)
echo "W4 swap post-processing job $PID (starts after all fit tasks end)"
