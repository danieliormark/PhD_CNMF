#!/bin/bash --login
#SBATCH --job-name=inc_p6
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing/logs_inc_p6/p6_%x_%A_%a.out
#
# Incremental run step 3 (RL-165): P6 v2 coreference on one unit, via inc_p6.py (coref_resolution_v2.py unchanged; v3 matcher).
#   New articles (15,563):        sbatch --array=0-99 --export=ALL,UNIT=new,NSHARDS=100 submit_inc_p6.sh
#   Current with additions (5,571): sbatch --array=0-39 --export=ALL,UNIT=current,NSHARDS=40 submit_inc_p6.sh
#   A failed or timed-out task: resubmit with the same exports and --array=<task ids>; every task runs with --resume.
#   When all tasks of a unit are done (on incline, PYTHONPATH=$HOME/np1_for_spacy):
#     python inc_p6.py --unit new --merge --nshards 100 --outdir /mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/inc_run/p6_new/
#     python inc_p6.py --unit current --merge --nshards 40 --outdir /mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/inc_run/p6_cur/
R=/mnt/hum01-rds/Basov/p91688di
case "$UNIT" in new) OUT=$R/llm_corpus_staging/inc_run/p6_new ;; current) OUT=$R/llm_corpus_staging/inc_run/p6_cur ;; *) echo "UNIT must be new or current"; exit 1 ;; esac
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/np1_for_spacy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "unit $UNIT task $SLURM_ARRAY_TASK_ID of $NSHARDS on $(hostname) at $(date -Is); coref sha $(sha256sum $R/pmc_preprocessing/coref_resolution_v2.py | cut -c1-12); wrapper sha $(sha256sum $R/pmc_preprocessing/inc_p6.py | cut -c1-12)"
python -u $R/pmc_preprocessing/inc_p6.py --unit $UNIT --shard $SLURM_ARRAY_TASK_ID --nshards $NSHARDS --outdir $OUT/ --resume
echo "exit $? at $(date -Is)"
