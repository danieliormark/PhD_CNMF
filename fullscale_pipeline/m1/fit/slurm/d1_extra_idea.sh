#!/bin/bash
# Idea coherence of the stage-1 fits (fit/d1_extra.py idea; RL-429): ≈ 4,500 fits × ≈ 1.5 s plus one content set per
# (approach, hold-out group) ≈ 2–3 CPU-hours; 16 workers, ≈ 20–30 min wall. Writes fit/results/stage1/[B_]d1_idea_coherence.json.
# Run from a CSF3 login node after stage 1 has finished:
#   bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/d1_extra_idea.sh          (study A)
#   STUDY=B bash .../d1_extra_idea.sh                                                                   (study B)
set -euo pipefail
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm
mkdir -p logs ../results/stage1
export WORKERS=16
if [ "${STUDY:-A}" = "B" ]; then
  export STAGE=B_stage1 BUILD=/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/ M1_STUDY=B
  export OUTFILE=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage1/B_d1_idea_coherence.json
else
  export STAGE=stage1 BUILD=
  export OUTFILE=/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage1/d1_idea_coherence.json
fi
JID=$(sbatch --parsable --cpus-per-task=$WORKERS --mem-per-cpu=4G --time=3:00:00 --export=ALL submit_d1_extra_idea.sh)
echo "D1 idea-coherence job $JID submitted ($STAGE → $OUTFILE)"
