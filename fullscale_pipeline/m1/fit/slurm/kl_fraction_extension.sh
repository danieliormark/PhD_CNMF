#!/bin/bash
# KL fraction grid extension (ticket 107, owner 2026-11-12): 18 fits (KL × 0.05/0.075/0.1 × K 5, 20 × 3 seeds), 3 array tasks
# × 6 fits, then the aggregation of the checkpoint-3 stage (it re-reads all checkpoint-3 fits). Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_kl_fraction_extension.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=6 STAGE=checkpoint3
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=6 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
AID=$(sbatch --parsable --dependency=afterok:$JID --time=02:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "fit array job $JID ($N fits, $TASKS tasks); aggregation job $AID (starts after all fit tasks succeed)"
