#!/bin/bash
# Checkpoint 3 on CSF3 (ticket 107; owner's permission required): 72 real-data fits (fraction check + timing, start
# comparison, float32), 9 array tasks × 8 fits, then the aggregation as a dependent job. Run from a CSF3 login node.
# Memory: the rehearsal used ~2 GB per fit at K = 3 (15.8–16.4 GB for a task of 8); K = 20 needs more, so 8 GB per core.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_checkpoint3.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=checkpoint3
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
AID=$(sbatch --parsable --dependency=afterok:$JID --time=02:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "fit array job $JID ($N fits, $TASKS tasks); aggregation job $AID (starts after all fit tasks succeed)"
