#!/bin/bash
# Timing test with the in-fit terms on (ticket 107, owner 2026-11-13): 28 fits (both losses × K 5, 20 × 7 arms: off,
# W regulariser, floor, floor with W, ceiling, O3, all), approach 1, hold-out group 1, seed 1, fraction 0.03, float32,
# each run for exactly 900 epochs (300 warm-up + 600 with terms). 4 array tasks × 7 fits. No aggregation (timing only;
# read with fit/timing_analysis.py). Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_timing_terms.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=7
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=7 --mem-per-cpu=8G --time=08:00:00 --export=ALL submit_stage.sh)
echo "timing array job $JID ($N fits, $TASKS tasks)"
