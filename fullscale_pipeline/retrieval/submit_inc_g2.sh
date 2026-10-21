#!/bin/bash --login
#SBATCH --job-name=inc_g2
#SBATCH --partition=multicore
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/logs_inc_g2/g2inc_%A_%a.out
#
# Incremental run step 5b (RL-168): G2 v3 via inc_g2.py (v3 hyphenated names kept, apostrophe re-parse) on the 120,582
# sentences of LCS/inc_run/g2_in/focal_sentences_inc.jsonl, 20 array tasks, output PG/g2_v3_inc/ (PG/g2_v3/ untouched).
#   sbatch /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/submit_inc_g2.sh
#   a failed task: sbatch --array=<i> ... (it resumes by itself)
NSHARDS=20
R=/mnt/hum01-rds/Basov/p91688di
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/np1_for_spacy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname) at $(date -Is); g2 sha $(sha256sum $R/phase5_graphbrain/scripts/graphbrain_parse_v3.py | cut -c1-12); wrapper sha $(sha256sum $R/phase5_graphbrain/scripts/inc_g2.py | cut -c1-12)"
python -u $R/phase5_graphbrain/scripts/inc_g2.py --shard $SLURM_ARRAY_TASK_ID --nshards $NSHARDS \
    --input $R/llm_corpus_staging/inc_run/g2_in/focal_sentences_inc.jsonl --outdir $R/phase5_graphbrain/g2_v3_inc/
echo "exit $? at $(date -Is)"
