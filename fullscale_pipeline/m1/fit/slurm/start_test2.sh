#!/bin/bash
# Start test 2 (ticket 107; owner 2026-11-19): room for links between different communities. KL, K 13, hold-out group 1,
# seeds 1–10; arms S1 (article×article random), S2 (social off-diagonal 5/10/20 %), S3 (as S2 + semantic) — 130 fits
# = 17 tasks × 8 fits; a KL K 13 fit took 30–55 min in the start test (ceiling 8,000 epochs ≈ 3 h), 12 h per task.
# Plan and expected outcomes: plans/ticket107_start_test2.md. Afterwards on incline:
#   python -m fit.align --stages start_test2
#   python -m fit.start_test2_analysis --out fit/results/stage0/start_test2.json
# Run from a CSF3 login node with bash (this launcher calls sbatch itself):
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/start_test2.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_start_test2.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=start_test2
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
