#!/bin/bash
# Binary test (ticket 107; owner 2026-11-16): fits on data where every visible tie value is 1 (arm binary), terms off,
# approach 1, each loss at K* (squared error 20, KL 13), hold-out group 1, seeds 1–3; 6 fits in 1 task × 8 CPUs, 12 h.
# Compared with the pass-1 weighted λ = 0 fits of the same seed by fit/w_form_diagnostics.py (incline). Run from a CSF3
# login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_binary_test.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=binary_test
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS task)"
