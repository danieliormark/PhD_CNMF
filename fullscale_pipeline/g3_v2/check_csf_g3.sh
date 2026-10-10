#!/bin/bash --login
#SBATCH --job-name=g3_check
#SBATCH --partition=serial
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/logs_g3/g3_check_%j.out
#
# Stage G3: test on a CSF compute node before the production run (G3_POSTPROCESSING.md §8 item 2).
#   sbatch /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/check_csf_g3.sh
# Checks the script hashes, runs the self-test and the hard cases, curates the whole of G2 v3 shards 0 and 20 into
# PG/g3_v1_csftest/job_<jobid>/ and compares every output record file byte for byte with the incline reference run of
# the same script (PG/g3_v1_csftest/reference_incline/, RUN_LOG RL-123). The last line must read "ALL G3 CHECKS OK".
# Tracked copy: PhD_CNMF/fullscale_pipeline/g3_v2/check_csf_g3.sh.

PG=/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain
D=$PG/scripts/g3_v2
REF=$PG/g3_v1_csftest/reference_incline
T=$PG/g3_v1_csftest/job_${SLURM_JOB_ID:-manual}
echo "== node: $(hostname)  cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)  date: $(date -Is)  job: ${SLURM_JOB_ID:-none}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tensor_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/np1_for_spacy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -o pipefail
fail() { echo "PROBLEM: $1"; echo "SOME G3 CHECKS FAILED"; exit 1; }

echo; echo "== 1. script hashes"
for x in "g3_curation_v2.py b3eadb6d385c" "modal_merge.py 75c7022ac187" "run_hard.py 80b9ca527b8f"; do
    set -- $x; got=$(sha256sum $D/$1 | cut -c1-12)
    echo "$1 $got"; [ "$got" = "$2" ] || fail "$1 sha256 $got, expected $2"
done

echo; echo "== 2. self-test"
OUT=$(cd $D && python g3_curation_v2.py --selftest 2>&1 | grep -E "FAIL|passed")
echo "$OUT"
echo "$OUT" | grep -q "^54 of 54 passed" || fail "self-test"

echo; echo "== 3. hard cases (expected: TOTAL 175/188, the 13 known failures)"
OUT=$(cd $D && python run_hard.py hard/hard_*.json 2>&1 | grep -E "pass|TOTAL")
echo "$OUT"
echo "$OUT" | grep -q "^TOTAL 175/188" || fail "hard cases"

echo; echo "== 4. shards 0 and 20, whole"
for i in 0 20; do
    (cd $D && python g3_curation_v2.py --shard $i --limit 0 --outdir $T 2>&1 | tail -n 1) || fail "shard $i (exit code)"
done

echo; echo "== 5. outputs against the incline reference"
for f in g3_test g3_noparent g3_articles g3_nonprose g3_errors; do
    for i in 000 020; do
        cmp -s $T/${f}_$i.jsonl $REF/${f}_$i.jsonl && echo "same   ${f}_$i.jsonl" || fail "${f}_$i.jsonl differs from the reference"
    done
done
python - $T <<'EOF' || fail "reports"
import json, sys
for i in ("000", "020"):
    r = json.load(open(f"{sys.argv[1]}/g3_report_{i}.json"))
    print(f"report {i}: problems {r['n_problems']}, unit errors {r['counts'].get('unit_errors', 0)}, units {r['counts']['units']}, "
          f"{r['seconds_per_unit']} s per unit")
    assert r["n_problems"] == 0 and not r["counts"].get("unit_errors")
EOF

echo; echo "ALL G3 CHECKS OK   (test output in $T; it can be deleted)"
echo "== done $(date -Is)"
