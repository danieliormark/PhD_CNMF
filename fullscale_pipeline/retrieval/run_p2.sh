#!/bin/bash
# Stage A, P2 only (rerun after the module-swap fix, RL-157)
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/retrieval || exit 1
G=/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/gap_stage
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh && conda activate tensor_env
PYTHONPATH=$HOME/np1_for_spacy python -u gap_stage_a.py p2 --workers 16 > $G/p2.log 2>&1 || { echo "p2 failed $(date)" >> $G/chain.log; exit 1; }
echo "p2 done $(date)" >> $G/chain.log
