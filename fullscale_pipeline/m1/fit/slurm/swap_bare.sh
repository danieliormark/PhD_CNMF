#!/bin/bash
# Bare graded cross-domain swap (ticket 107; decided 2026-11-08/09, run alongside pass 2 by owner decision 2026-11-16).
# A fraction p ∈ {0.25, 0.5, 0.75, 1} of articles has its identities permuted, within social-degree bands, in the social
# relations only; terms off; each loss at K* (squared error 20, KL 13), both approaches, hold-out group 1, seeds 1–10;
# p = 0 = pass 1. 160 fits = 20 tasks × 8 fits, 24 h per task. Read afterwards on incline by fit/swap_analysis.py.
# Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_swap_bare.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=swap_bare
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
