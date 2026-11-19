#!/bin/bash
# Stage 0 pass 3, study A (genre hyperedges in; ticket 107, owner 2026-11-19: "I approve the sizing, please proceed with
# the plan"): K ∈ {5, 8, 13, 20} (RL-296) × r ∈ {0, 0.3, 1} (RL-298; λ_child = r × the pass-1 basis of that K, RL-297) ×
# approaches 1, 2 × hold-out groups 1–5 × seeds 1–10 = 1,200 fits; KL, start S3 at 10 % (RL-280), fraction 0.03, float32,
# warm-up 300, 8,000-epoch ceiling. 150 array tasks × 8 fits (one thread each), 12 h per task (longest fits, K 20 with
# the term, ≈ 2.5 h); ≈ 850–900 CPU-hours. Then, after every fit task ends: alignment, aggregation (as fitted and
# aligned), readings and the J1 choice (RL-299) → fit/results/stage0/pass3_{choice,readings}_A.{json,log}.
# Re-submitting skips finished fits. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/stage0_pass3.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_pass3.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass3 STUDY=A
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=24:00:00 --mem-per-cpu=8G --export=ALL submit_pass3_post.sh)
echo "pass-3 post-processing job $PID (alignment, aggregation, readings, J1 choice; starts after all fit tasks end)"
