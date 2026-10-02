# Words and sentences per article, summed over ALL region files (<PMCID>.txt, .r2.txt, ...),
# same naive counting as before (whitespace words; regex sentence split).
import json, os, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics")
from fullscale_analysis import SENT_SPLIT
C = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/clean_corpus_v2/"
files = collections.defaultdict(list)
for f in os.listdir(C):
    files[f.split(".")[0]].append(f)
def one(p):
    w = s = 0
    for f in files[p]:
        t = open(C + f, encoding="utf-8", errors="replace").read().strip()
        w += len(t.split()); s += len(SENT_SPLIT.split(t)) if t else 0
    return p, dict(words=w, sentences=s, n_files=len(files[p]))
with ThreadPoolExecutor(32) as ex:
    res = dict(ex.map(one, list(files)))
json.dump(res, open("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/wordcount_allregions.json", "w"))
print("articles", len(res), "files", sum(v["n_files"] for v in res.values()), "words", sum(v["words"] for v in res.values()), "sentences", sum(v["sentences"] for v in res.values()))
