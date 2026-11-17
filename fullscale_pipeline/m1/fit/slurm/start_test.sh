#!/bin/bash
# Diagonal-start test (ticket 107; owner 2026-11-17). Hybrid start of pass 1 with every social core matrix started
# diagonal (init.diagonal_social_links); each loss at K* (squared error 20, KL 13), both approaches, hold-out group 1,
# seeds 1–10, terms off; reference = the pass-1 fits of the same group and seeds. 40 fits = 5 tasks × 8 fits, 24 h per
# task. Expected outcomes and reading rule: plans/ticket107_start_test.md. Read on incline afterwards:
#   python -m fit.start_test_analysis --out fit/results/stage0/start_test.json
# Run from a CSF3 login node with bash (this launcher calls sbatch itself):
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/start_test.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_start_test.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=start_test
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
