#!/bin/bash
# Stage 1, study A (ticket 107): D1 journal arms J1, J2, J3a, J3b, J4, J5 and the robustness arms (J2 + social log-damping;
# semantic share 0.65 at J1 and J2) × approaches 1, 2 × coarse K 3, 5, 8, 13, 20 × hold-out groups 1–10 × seeds 1–5
# (owner decisions RL-383/385/389/402/403; λ_child = r*(approach, K) × study A's pass-1 basis, rule C with amendment,
# shared by every arm at a K); KL, start S3 at 10 %, fraction 0.03, float32, warm-up 300, 8,000-epoch ceiling.
# 4,500 fits ≈ 2,200 CPU-hours of fitting; 563 array tasks × 8 fits (one thread each, tasks are homogeneous in K),
# 12 h per task. Then, after every fit task ends: alignment and aggregation (as fitted and aligned). The D1 analysis
# (D1-M1..M4, decision rule) and the robustness readings are separate scripts, written before the results are read.
# Re-submitting skips finished fits. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/stage1.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage1.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage1
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=8G --export=ALL submit_stage1_post.sh)
echo "stage-1 post-processing job $PID (alignment, aggregation; starts after all fit tasks end)"
