#!/bin/bash --login
#SBATCH --job-name=g3_v1
#SBATCH --partition=serial
#SBATCH --array=0-49
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/logs_g3/g3_v1_%A_%a.out
#
# Stage G3 production (G3_POSTPROCESSING.md §8 item 2): curates one G2 v3 shard per array task with g3_curation_v2.py,
# the same script checked by check_csf_g3.sh (RL-123). No parser is loaded (G2 v3's own parse and lemma edges are
# reused); each task reads only its own shard file and needs no shared state with the others.
#   sbatch /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/submit_g3_v1.sh
#   one failed task: g3_curation_v2.py refuses to overwrite, so first delete its 6 files in PG/g3_v1/shards/ for that
#   shard number (g3_{test,noparent,articles,nonprose,errors}_NNN.jsonl, g3_report_NNN.json), then
#     sbatch --array=<i> /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/submit_g3_v1.sh
# After all 50 tasks:
#   python /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/merge_g3_v1.py --nshards 50
# G2 v3 output (PG/g2_v3/) is not touched. Tracked copy: PhD_CNMF/fullscale_pipeline/g3_v2/submit_g3_v1.sh.

D=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2
OUT=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/shards/
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/np1_for_spacy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date -Is)"
mkdir -p $OUT
cd $D && python g3_curation_v2.py --shard $SLURM_ARRAY_TASK_ID --limit 0 --outdir $OUT
echo "exit $? at $(date -Is)"
