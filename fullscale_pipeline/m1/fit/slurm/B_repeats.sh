#!/bin/bash
# Study B (no genre hyperedges, RL-293): repeats of study A's KL runs of stage 0 (owner 2026-11-19, RL-291) at K 13 in both
# approaches (owner 2026-11-20, RL-304): pass 2 (r 0.03 / 0.1 / 0.3 / 1; λ_child = r × B's basis at K 13: 56 / 187 / 562 /
# 1,875 approach 1, 52 / 173 / 518 / 1,728 approach 2; 400 fits), bare swap and follow-ups (120), diagonal-start test
# (20), damping test (18), start test 2 (130) — 688 fits, each identical to A's job except stage "B_…", build and λ.
# 86 array tasks × 8 fits, 12 h per task (K 13 ≈ 0.7–1 h, K 20 follow-ups ≈ 2 h); ≈ 600 CPU-hours. Then alignment of
# all study-B stages and the aggregation of B's pass 2. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/B_repeats.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_B_repeats.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8
export BUILD=/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=12:00:00 --mem-per-cpu=8G --export=ALL submit_B_repeats_post.sh)
echo "study-B repeats post-processing job $PID (alignment, pass-2 aggregation; starts after all fit tasks end)"
