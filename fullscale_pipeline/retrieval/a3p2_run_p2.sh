#!/bin/bash
# A3 part 2, step 1 (RL-203): P2 with focal_terms_v3_1 (v3 + G3's hub spellings) on the current corpus and on the gap
# corpus, into separate folders (existing P2 outputs untouched). Same launch as p2_v3_current/run.sh (RL-165).
cd /mnt/hum01-rds/Basov/p91688di/pmc_preprocessing || exit 1
source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh && conda activate tensor_env
A=/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/a3p2
for UNIT in current gap; do
PYTHONPATH=$HOME/np1_for_spacy UNIT=$UNIT python -u - > $A/p2_$UNIT/p2.log 2>&1 <<'PY'
import os, sys, importlib.util
R = "/mnt/hum01-rds/Basov/p91688di/"; PP = R + "pmc_preprocessing/"; LCS = R + "llm_corpus_staging/"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
load("focal_terms", PP + "focal_terms.py")
load("focal_terms_v3", PP + "focal_terms_v3.py")
sys.modules["focal_terms"] = load("focal_terms_v3_1", PP + "focal_terms_v3_1.py")
m = load("p2", PP + "focal_extraction_v2.py")
unit = os.environ["UNIT"]
if unit == "gap":                                     # the staging tree of the gap corpus, as gap_stage_a.point()
    GR = LCS + "gap_stage/"; m.P, m.LCS = GR, GR + "llm_corpus_staging/"
    for attr in ("RAW", "RAW_DIR"):
        if hasattr(m, attr): setattr(m, attr, LCS + "pure_text_corpus_gap/")
    m.CLEAN = GR + "llm_corpus_staging/clean_corpus_v2/"
sys.argv = ["p2", "--outdir", LCS + "a3p2/p2_" + unit + "/", "--workers", "24"]
m.main()
PY
echo "$UNIT exit $? $(date)" >> $A/done.txt
done
