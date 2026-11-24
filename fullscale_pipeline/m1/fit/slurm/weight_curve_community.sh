#!/bin/bash
# Community-quality readings across the weighting points (owner 2026-11-24: "Yes, please prepare the job"; RL-362):
# W1 / 0.75 / 0.65 / W2 at K 5, r 0.3, study A, both approaches, 400 saved fits; no new fits. One job, 8 cores, ≈ 15–30
# min → fit/results/stage0/weight_curve_community.{json,log}. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/weight_curve_community.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
JID=$(sbatch --parsable --cpus-per-task=8 --mem-per-cpu=4G --time=03:00:00 --export=ALL submit_weight_curve_community.sh)
echo "community readings job $JID submitted"
