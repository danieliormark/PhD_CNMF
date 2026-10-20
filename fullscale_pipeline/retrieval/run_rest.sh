#!/bin/bash
# Stage A, steps P1b -> P1c -> P2 (RL-157); each step only if the previous one succeeded
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/retrieval || exit 1
G=/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/gap_stage
python3 -u gap_stage_a.py p1b --workers 24 > $G/p1b.log 2>&1 || { echo "p1b failed" >> $G/chain.log; exit 1; }
echo "p1b done $(date)" >> $G/chain.log
python3 -u gap_stage_a.py p1c --workers 24 > $G/p1c.log 2>&1 || { echo "p1c failed" >> $G/chain.log; exit 1; }
echo "p1c done $(date)" >> $G/chain.log
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh && conda activate tensor_env
PYTHONPATH=$HOME/np1_for_spacy python -u gap_stage_a.py p2 --workers 16 > $G/p2.log 2>&1 || { echo "p2 failed" >> $G/chain.log; exit 1; }
echo "p2 done $(date)" >> $G/chain.log
