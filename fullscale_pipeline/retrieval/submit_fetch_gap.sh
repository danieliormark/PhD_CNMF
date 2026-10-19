#!/bin/bash --login
#SBATCH --job-name=pmc_gap_fetch
#SBATCH --output=/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/slurm-gap-%j.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH -p serial
# R3b (RL-154): full texts of the retrieval-gap candidates; see fetch_gap_pmc.py
source activate tensor_env
python3 -u /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/retrieval/fetch_gap_pmc.py /mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/gap_pmcids_20261019_tiab.txt
