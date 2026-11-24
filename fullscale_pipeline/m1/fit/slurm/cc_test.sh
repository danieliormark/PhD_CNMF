#!/bin/bash
# child×child focused test before stage 1 (ticket 107; owner 2026-11-24: "I approve the focused test before stage 1",
# design decided RL-377): W fitted as a child×child relation (full / ties-only × weight matched to r 0.3 / 1) and the
# combined arm (full at r 0.3 + regulariser r 0.3), approach 1, K 5 and 13, groups 1–5 × seeds 1–10, settings as pass 3;
# 500 fits ≈ 400 CPU-hours; 63 tasks × 8 fits, 12 h. Then alignment, aggregation and the analysis with the decision
# rule fixed before the results → fit/results/stage0/cc_test.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/cc_test.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_cc_test.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --cpus-per-task=8 --time=06:00:00 --mem-per-cpu=6G --export=ALL submit_cc_test_post.sh)
echo "child×child test post-processing job $PID (starts after all fit tasks end)"
