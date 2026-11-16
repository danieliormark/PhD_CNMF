#!/bin/bash
# Stage 0 pass 2, highest-grid-value rule for KL (ticket 107; decided 2026-11-14, fired in pass 2 (RL-244); owner
# 2026-11-16: "I approve the KL r = 1 extension"): KL at r = 1 (λ_child = r × data loss ÷ (R ÷ S) at λ = 0), contrast
# form, K* = 13, both approaches, hold-out groups 1–5 × 10 seeds; settings as pass 2. 100 fits = 13 tasks × ≤ 8 fits,
# 24 h per task, then the aggregation of the whole pass-2 stage (all r). Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_pass2_kl_r1.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass2
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=24:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
AID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "aggregation job $AID (starts after all fit tasks end)"
