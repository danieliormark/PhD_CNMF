#!/bin/bash
# r sweep of the membership readings (owner 2026-11-23: "I approve making this short test"; RL-357): R1–R5 on every
# (approach, K, r) of study A — pass 3 (r 0 / 0.3 / 1) and the λ calibration (three more r per K) — to separate the
# word-similarity term's strength r from K in the child hyperedges' instability, with membership spread per tie band.
# Saved fits only. 60 cells × 5 groups = 300 tasks; one job, 8 cores, ≈ 1 h. Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/membership_readings_r_sweep.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
JID=$(sbatch --parsable --cpus-per-task=8 --mem-per-cpu=4G --time=06:00:00 --export=ALL,MR_SET=r_sweep submit_membership_readings.sh)
echo "membership readings (r sweep) job $JID submitted"
