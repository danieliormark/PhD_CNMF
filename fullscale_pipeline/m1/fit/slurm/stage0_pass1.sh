#!/bin/bash
# Stage 0, pass 1 (ticket 107; owner 2026-11-13: S = 10 seeds, "please proceed with the plan"): 1,000 fits = approaches
# 1 and 2 × squared error and KL × K 3, 5, 8, 13, 20 × hold-out groups 1–5 × 10 seeds; in-fit terms off; fraction 0.03;
# float32; hybrid start; up to 5,000 epochs. 125 array tasks × 8 fits (one thread each), then the aggregation (K
# selection per loss and approach), which starts when every fit task has ended (afterany: it also lists unfinished fits).
# Estimated ≈ 740 CPU-hours; the longest fits (K = 20) ≈ 2–3 h. Re-submitting skips finished fits. Run from a CSF3 login node.
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs
export JOBS=$PWD/jobs_stage0_pass1.jsonl OUT=/mnt/hum01-rds/Basov/p91688di/m1_fits PER_TASK=8 STAGE=stage0_pass1
N=$(wc -l < "$JOBS"); TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
# If the fit array is already queued, pass its id to submit only the aggregation:  FIT_JOB=<id> bash stage0_pass1.sh
if [ -n "${FIT_JOB:-}" ]; then
  JID=$FIT_JOB
else
  JID=$(sbatch --parsable --array=0-$((TASKS - 1)) --cpus-per-task=8 --mem-per-cpu=8G --time=12:00:00 --export=ALL submit_stage.sh)
  echo "fit array job $JID submitted"
fi
# multicore allows at most 8 GB per core; submit_aggregate.sh asks for 2 cores → 16 GB
AID=$(sbatch --parsable --dependency=afterany:$JID --time=06:00:00 --mem-per-cpu=8G --export=ALL submit_aggregate.sh)
echo "stage-0 pass-1 fit array job $JID ($N fits, $TASKS tasks); aggregation job $AID (starts after all fit tasks end)"
