#!/bin/bash
# P2 rerun 3 (group-4 guard after the Stage B hand check) then Stage B (RL-157)
G=/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/gap_stage
$G/run_p2.sh || exit 1
cd /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/retrieval || exit 1
python3 -u gap_stage_b.py --workers 24 > $G/stage_b.log 2>&1 && echo "stage_b done $(date)" >> $G/chain.log || echo "stage_b failed $(date)" >> $G/chain.log
