#!/bin/bash
# D1 analysis of stage 1 (ticket 107; fit/d1_analysis.py, written 2026-11-26 before any stage-1 result was read).
# Reads the finished stage-1 fits on RDS (converged fits only; numbering-free readings, so it does not need the
# alignment step) and writes fit/results/stage1/[B_]d1_analysis.json (+ the log). ≈ 100 units (approach × K × group),
# measured ≈ 4 min per 15-fit K 20 unit before the co-membership speed-up; estimate ≈ 8–10 CPU-hours for study A on
# 16 workers (≈ 40 min wall time). Run from a CSF3 login node after stage 1 has finished:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/d1_analysis.sh        (study A)
#   STUDY=B bash .../d1_analysis.sh                                                                  (study B)
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs ../results/stage1
export WORKERS=16
if [ "${STUDY:-A}" = "B" ]; then
  export STAGE=B_stage1 BUILD=/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/ M1_STUDY=B
  export OUTFILE=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage1/B_d1_analysis.json
else
  export STAGE=stage1 BUILD=
  export OUTFILE=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage1/d1_analysis.json
fi
JID=$(sbatch --parsable --cpus-per-task=$WORKERS --mem-per-cpu=6G --time=8:00:00 --export=ALL submit_d1_analysis.sh)
echo "D1 analysis job $JID submitted ($STAGE → $OUTFILE)"
