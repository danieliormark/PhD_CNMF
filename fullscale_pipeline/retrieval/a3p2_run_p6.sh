#!/bin/bash
# A3 part 2, step 3 (RL-203): P6 v2 (a3p2_p6.py, coref unchanged, v3.1 matcher) on the 180 articles with an addition,
# locally on incline: current 8 shards, gap 2 shards, 3 threads each; then the merges.
R=/mnt/hum01-rds/Basov/p91688di; A=$R/llm_corpus_staging/a3p2
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONPATH=$HOME/np1_for_spacy
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $R/pmc_preprocessing
for k in 0 1 2 3 4 5 6 7; do python -u a3p2_p6.py --unit current --shard $k --nshards 8 --outdir $A/p6_current/ --resume > $A/p6_current_$k.log 2>&1 & done
for k in 0 1; do python -u a3p2_p6.py --unit gap --shard $k --nshards 2 --outdir $A/p6_gap/ --resume > $A/p6_gap_$k.log 2>&1 & done
wait
python -u a3p2_p6.py --unit current --merge --nshards 8 --outdir $A/p6_current/ > $A/p6_current_merge.log 2>&1
python -u a3p2_p6.py --unit gap --merge --nshards 2 --outdir $A/p6_gap/ > $A/p6_gap_merge.log 2>&1
echo "p6 done $(date)" >> $A/done.txt
