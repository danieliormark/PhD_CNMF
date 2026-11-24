#!/bin/bash
# Item (l) sufficient members on stage-0 fits (owner 2026-11-24, RL-371/372): member counts per community, domain and
# facet at q 0.2 / 0.3 / 0.4 / 0.5 among covered entities, robust minimum (log counts, median − k × 1.4826 MAD, k 2.5 /
# 2, zero always fails, majority of seeds), articles counted as social and not counted; pass-3 candidate cells (W1) of
# studies A and B, 750 saved fits; one job, 8 cores, ≈ 10–20 min → fit/results/stage0/sufficient_members.{json,log}.
# Run from a CSF3 login node:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/sufficient_members.sh
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
JID=$(sbatch --parsable --cpus-per-task=8 --mem-per-cpu=4G --time=02:00:00 --export=ALL submit_sufficient_members.sh)
echo "sufficient members job $JID submitted"
