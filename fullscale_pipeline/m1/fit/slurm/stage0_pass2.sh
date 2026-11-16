#!/bin/bash
# Stage 0 pass 2 (ticket 107): λ_child per loss, contrast form R ÷ S (option B, owner 2026-11-16). Each loss at its K*,
# λ_child = r × (data loss ÷ (R ÷ S)) at λ = 0 (fit/pass2_lambda.py → results/stage0/pass2_lambda_basis_v2.json),
# r: squared error {0.003, 0.01, 0.03}, KL {0.03, 0.1, 0.3}; hold-out groups 1–5 × 10 seeds; fraction 0.03, float32, hybrid start, warm-up 300, 8,000-epoch
# ceiling. 600 fits = 75 tasks × 8 fits, 24 h per task, then the aggregation of the pass-2 stage.
# Owner decisions 2026-11-14 (scaling, grid, K*, permission to write the job). Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_pass2.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass2
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
AID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "aggregation job $AID (starts after all fit tasks end)"
