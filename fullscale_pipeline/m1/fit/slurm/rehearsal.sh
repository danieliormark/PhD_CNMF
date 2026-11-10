#!/bin/bash
# CSF3 rehearsal (ticket 107, F8): run from a CSF3 login node. Submits the 17-fit array (3 tasks × up to 8 fits) and the
# aggregation as a dependent job. Afterwards: sacct -j <ids> --format=JobID,State,Elapsed,MaxRSS,NodeList
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_rehearsal.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=rehearsal
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=4G --time=01:00:00 --export=ALL submit_stage.sh)
AID=$(sbatch --parsable --dependency=afterok:$JID --time=00:30:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "fit array job $JID ($N fits, $TASKS tasks); aggregation job $AID (starts after all fit tasks succeed)"
