#!/bin/bash
# Intermediate weighting check (ticket 107; owner 2026-11-23: "I approve option (a), prepare the intermediate weighting
# check"; RL-353): the semantic domain's share of the KL pull set to 0.65 and 0.75 (mass weights, then each domain × 2s /
# 2(1 − s); split inside a domain and total pull unchanged), between W2 (0.50, W3 check) and W1 (0.83 / 0.90, pass 3).
# Study A, both approaches, K 5, r 0.3, groups 1–5 × seeds 1–10 (200 fits) + the swap at p = 1 (K 5, r 0.3, group 1,
# seeds 1–10) at all four points (80 fits); settings as pass 3. 280 fits at K 5 ≈ 100–150 CPU-hours; 35 tasks × 8 fits,
# 12 h. Then alignment, aggregation and the curve → fit/results/stage0/weight_curve.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/weight_curve.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_weight_curve.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_weight_curve_post.sh)
echo "weight-curve post-processing job $PID (starts after all fit tasks end)"
