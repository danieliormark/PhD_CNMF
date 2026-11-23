#!/bin/bash
# Item (l) option (a) (owner 2026-11-23: "I approve option a, prepare the readings to avoid this self-circularity problem
# you raised"; RL-351): per-entity membership agreement across seeds (R1, Track A per entity), alignment-free
# co-membership agreement (R2), placement without a seed-variability term (R3), the circularity check by seed halves
# (R4) and inspection flags (R5), on the pass-3 candidate cells of studies A and B (15 cells × 5 hold-out groups = 75
# tasks, saved fits only, no new fits). Test task: ~3 min, 1.1 GB. One job, 8 cores.
# Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/membership_readings.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
JID=$(sbatch --parsable --cpus-per-task=8 --mem-per-cpu=4G --time=04:00:00 submit_membership_readings.sh)
echo "membership readings job $JID submitted"
