#!/bin/bash
# W3 check (ticket 107; owner 2026-11-22: "I approve W3, prepare the check"; RL-339/340): KL with mass weights (each
# relation's pull = w_r × T, so the decided 50 / 50 domain split holds under KL; total pull unchanged) vs the current
# Frobenius scaling (semantic 83–90 %), paired with pass 3's fits. Study A, both approaches, K 5 (r 0.3) and K 13 (r 1),
# groups 1–5 × seeds 1–10 (200 fits) + the swap at p = 1 (K 13, r 1, group 1, seeds 1–10) under both scalings (40 fits);
# settings as pass 3. 240 fits ≈ 150–200 CPU-hours; 30 tasks × 8 fits, 12 h. Then alignment, aggregation and the
# comparison → fit/results/stage0/weight_check.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/weight_check.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_weight_check.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_weight_check_post.sh)
echo "W3 post-processing job $PID (starts after all fit tasks end)"
