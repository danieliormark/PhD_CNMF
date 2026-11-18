#!/bin/bash
# Equivalence test of the social log-damping arm (ticket 107; owner 2026-11-18). KL, K = 13, both approaches, hold-out
# group 1, seeds 1–3, diagonal social start, terms off; arms weight / data_matched / data (fit/plan.py damping_test).
# 18 fits = 3 tasks × 6 fits; a KL K = 13 fit took 30–55 min in the start test (8,000-epoch ceiling ≈ 3 h), 8 h per
# task. Plan and expected outcomes: plans/ticket107_damping_test.md. Read on incline afterwards:
#   python -m fit.damping_test_analysis --out fit/results/stage0/damping_test.json
# Run from a CSF3 login node with bash (this launcher calls sbatch itself):
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/damping_test.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_damping_test.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=6 STAGE=damping_test
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=6 --mem-per-cpu=8G --time=08:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
