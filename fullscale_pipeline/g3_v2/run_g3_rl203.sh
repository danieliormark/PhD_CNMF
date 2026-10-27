#!/bin/bash
# RL-203: G3 production rerun on G2 v3.3 (A3 part 2), 50 shards on incline, 10 in parallel (as RL-185/186); then merge.
D=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2; OUT=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v2/shards/
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONPATH=$HOME/np1_for_spacy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $D
seq 0 49 | xargs -P 10 -I{} sh -c "python g3_curation_v2.py --shard {} --limit 0 --show 0 --outdir $OUT > $OUT/../run_{}.log 2>&1; echo shard {} exit \$?" 
python merge_g3_v2.py --nshards 50 > $OUT/../merge.log 2>&1; echo "merge exit $?"
