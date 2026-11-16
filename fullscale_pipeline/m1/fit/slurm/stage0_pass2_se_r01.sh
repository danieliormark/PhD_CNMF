#!/bin/bash
# Stage 0 pass 2, squared error at r = 0.1 (ticket 107; owner 2026-11-16: "I approve your proposal b for the values of
# r"): squared error chose its highest grid value r = 0.03 in pass 2 (RL-244). r = 0.1 (λ_child = r × data loss ÷ (R ÷ S)
# at λ = 0: 0.146 approach 1, 0.141 approach 2), contrast form, K* = 20, both approaches, hold-out groups 1–5 × 10 seeds;
# settings as pass 2. 100 fits = 13 tasks × ≤ 8 fits, 24 h per task (pass-2 squared-error fits: mean 2.0 h, max 4.8 h),
# then the aggregation of the whole pass-2 stage (all r). Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_pass2_se_r01.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass2
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
AID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "aggregation job $AID (starts after all fit tasks end)"
