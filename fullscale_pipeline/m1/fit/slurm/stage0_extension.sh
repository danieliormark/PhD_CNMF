#!/bin/bash
# Stage 0 pass 1 extension (ticket 107; owner 2026-11-14 decisions 1–2): the pre-registered K = 32 (200 fits: approaches
# 1, 2 × both losses × groups 1–5 × 10 seeds) and the rerun of the 13 squared-error approach-1 K = 13 fits that reached
# the 5,000-epoch ceiling; 8,000-epoch ceiling; fraction 0.03, float32, hybrid start, terms off. 27 tasks × 8 fits,
# 24 h per task (K = 32 ≈ 3.2 s per epoch → ≤ 7 h per fit), then the aggregation of the whole stage (reruns replace their
# originals: aggregate.latest_fits). Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_extension.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass1
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
AID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "aggregation job $AID (starts after all fit tasks end)"
