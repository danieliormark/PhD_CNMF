#!/bin/bash --login
#SBATCH --job-name=g3_v2
#SBATCH --partition=serial
#SBATCH --array=0-49
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/logs_g3/g3_v2_%A_%a.out
#
# Stage G3 production rerun on the merged corpus (RL-171; copy of submit_g3_v1.sh, RL-124): curates one G2 v3.1 shard
# (PG/g2_v3_1/shards/) per array task with g3_curation_v2.py (D6, item 11, v3 matcher, the lists of PG/g3_inputs_v3_1/),
# the same script checked by check_csf_g3_v2.sh. No parser is loaded; each task reads only its own shard file.
#   sbatch /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/submit_g3_v2.sh
#   one failed task: g3_curation_v2.py refuses to overwrite, so first delete its 7 files in PG/g3_v2/shards/ for that
#   shard number (g3_{test,noparent,articles,nonprose,errors,occurrences}_NNN.jsonl, g3_report_NNN.json), then
#     sbatch --array=<i> /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/submit_g3_v2.sh
# After all 50 tasks:
#   python /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/merge_g3_v2.py --nshards 50
# G2 output (PG/g2_v3_1/) and the RL-130 G3 output (PG/g3_v1/) are not touched.
# Tracked copy: PhD_CNMF/fullscale_pipeline/g3_v2/submit_g3_v2.sh.

D=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2
OUT=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v2/shards/
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/np1_for_spacy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date -Is)"
mkdir -p $OUT
cd $D && python g3_curation_v2.py --shard $SLURM_ARRAY_TASK_ID --limit 0 --show 0 --outdir $OUT
echo "exit $? at $(date -Is)"
