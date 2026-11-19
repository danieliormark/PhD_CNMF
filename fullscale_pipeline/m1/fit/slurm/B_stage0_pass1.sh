#!/bin/bash
# Study B (build without genre hyperedges, RL-293), stage 0 pass 1 (owner 2026-11-19: "I approve repeating the KL
# runs", RL-291): study A's KL pass-1 jobs and its K = 32 extension on the study-B build — 600 fits = approaches 1, 2 ×
# K 3, 5, 8, 13, 20, 32 × hold-out groups 1–5 × 10 seeds; KL, fraction 0.03, float32, hybrid start, terms off;
# 5,000-epoch ceiling (8,000 at K = 32). Study B's own hold-out draw (seed 107; owner: fresh sampling). 75 array tasks
# × 8 fits (one thread each), then the aggregation with study B's build. Estimate from study A's same fits: ≈ 660
# CPU-hours; longest fits (K = 32) ≈ 3.6 h, at most ≈ 4.5 h. Re-submitting skips finished fits. Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_B_stage0_pass1.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=B_stage0_pass1
export BUILD=/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
AID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "study-B pass-1 aggregation job $AID (starts after all fit tasks end)"
