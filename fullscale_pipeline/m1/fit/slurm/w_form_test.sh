#!/bin/bash
# W-term remedies test (ticket 107; RL-234; owner 2026-11-15): option A (decided form) and option B (contrast form R ÷ S),
# both at r ∈ {0.001, 0.003, 0.01, 0.03, 0.1}, approach 1, each loss at K*, hold-out group 1, seeds 1–3; 60 fits = 8 tasks × 8,
# 12 h per task (pass-1 K* fits took 1.1–2.9 h; 8,000-epoch ceiling). No aggregation: the comparison with λ = 0 is
# fit/w_form_diagnostics.py, run afterwards on incline. Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_w_form_test.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=w_form_test
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
echo "fit array job $JID submitted ($N fits, $TASKS tasks)"
