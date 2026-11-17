#!/bin/bash
# Swap follow-ups at p = 1 (ticket 107; owner 2026-11-17): (i) one band for all articles at K*; (ii) bands at the next
# coarse K above K* (squared error 32, KL 20). Both approaches, both losses, hold-out group 1, seeds 1–10, terms off.
# 80 fits = 10 tasks × 8 fits, 24 h per task (K = 32 squared error is the slowest). Read on incline afterwards.
# Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_swap_followup.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=swap_bare
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
