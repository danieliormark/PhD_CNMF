#!/bin/bash
# Stage 0 pass 3, study B (no genre hyperedges; owner 2026-11-20: "Yes, prepare study B's pass 3"): the same protocol as
# study A (RL-296–313; C3 RL-295): K ∈ {3, 5, 8, 13, 20} (K 3 included from the start, as A's grid-end extension) ×
# r ∈ {0, 0.3, 1} (λ_child = r × study B's pass-1 basis of that K and approach, pass3_lambda_basis_B.json) × approaches
# 1, 2 × hold-out groups 1–5 × seeds 1–10 = 1,500 fits; KL, start S3 at 10 % (B's start test 2 gives the same choice,
# RL-314), fraction 0.03, float32, warm-up 300, 8,000-epoch ceiling. 188 array tasks × 8 fits, 12 h per task; ≈ 900–950
# CPU-hours. Then, after every fit task ends: alignment, aggregation (as fitted and aligned, B's build), readings and the
# J1 choice under option (b) → fit/results/stage0/pass3_{choice,readings}_B.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/B_stage0_pass3.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_B_stage0_pass3.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=B_stage0_pass3 STUDY=B
export BUILD=/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=24:00:00 --mem-per-cpu=8G --export=ALL submit_pass3_post.sh)
echo "study-B pass-3 post-processing job $PID (alignment, aggregation, readings, J1 choice; starts after all fit tasks end)"
