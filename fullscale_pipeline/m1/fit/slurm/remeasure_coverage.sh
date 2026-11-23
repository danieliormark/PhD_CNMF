#!/bin/bash
# Item (l) remeasurement (owner 2026-11-23: "I propose you start with the task of remeasuring the readings that used the
# placed rule, now under the coverage rule"; criterion RL-345): saved fits only, no new fits. Every reading that used
# the placed rule is recomputed with the coverage rule c = 0.5 (group-level: mean coverage over the group's seeds), with a
# per-fit coverage reading beside it in the aggregations; earlier outputs are kept as *_placed.*. 5 array tasks, 8 cores:
#   0  study A pass 3: aggregation (aligned) → J1 choice (stability gate, ghost and alignment vetoes, Track A, λ-rule guard)
#   1  study B pass 3: the same
#   2  start tests 1 and 2, both studies (start tie-break by stability)
#   3  stability tests 1–3 (study A vs B, stage-0 pass 1)
#   4  S3 gate 1 (excess spread vs held-out AUC, stage-0 pass 1, both approaches and losses)
# Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/remeasure_coverage.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
JID=$(sbatch --parsable --array=0-4 --cpus-per-task=8 --mem-per-cpu=6G --time=12:00:00 submit_remeasure_coverage.sh)
echo "remeasurement array job $JID submitted (5 tasks)"
