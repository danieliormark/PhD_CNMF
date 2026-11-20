#!/bin/bash
# Stage 0 pass 3, study A — grid-end extension (ticket 107; decided 2026-11-19: pass 3's grid is "extended if the new K*
# falls at the grid's end"): J1 chose K = 5, the lowest grid value, in both approaches (RL-307) → K = 3 × r ∈ {0, 0.3, 1}
# × approaches 1, 2 × groups 1–5 × seeds 1–10 = 300 fits, everything else as stage0_pass3.sh (λ_child = r × the pass-1
# basis at K 3: 2,501 / 2,299). Same stage name, so the post-processing (alignment, aggregation, readings, J1 choice)
# reads all 1,500 pass-3 fits. 38 tasks × 8 fits, 6 h per task (a K 3 fit ≈ 10–20 min); ≈ 60–80 CPU-hours.
# Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/stage0_pass3_K3.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_pass3_K3.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass3 STUDY=A
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=06:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
PID=$(sbatch --parsable --dependency=afterany:$JID --time=24:00:00 --mem-per-cpu=8G --export=ALL submit_pass3_post.sh)
echo "pass-3 post-processing job $PID (all pass-3 fits; starts after all fit tasks end)"
