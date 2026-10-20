# Probe (read-only): P0's own region rule on the downloaded gap articles; is a focal term (current matcher) inside the
# analysed regions? Rough P2 survival estimate: no citation masking or non-prose removal here.
import importlib.util, json, os, sys, collections
from multiprocessing import Pool
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
spec = importlib.util.spec_from_file_location("p0", "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing/build_content_regions.py")
p0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p0)
p0.RAW_DIR = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus_gap/"
from focal_terms import Matcher
M = Matcher()
def one(pid):
    try:
        row = p0.analyse(pid)
    except Exception as e:
        return pid, "p0_error", 0, 0
    lines = open(p0.RAW_DIR + pid + ".txt", encoding="utf-8", errors="replace").read().split("\n")
    regs = json.loads(row.get("regions_json") or "[]")
    txt = "\n".join("\n".join(lines[r["start"]:r["end"]]) for r in regs)
    whole = "\n".join(lines)
    anch_r = bool(M.anchor_hits(txt)); anch_w = bool(M.anchor_hits(whole))
    hr = sum(len(M.hits(l, anch_r)[0]) for l in txt.split("\n") if l.strip())
    hw = sum(len(M.hits(l, anch_w)[0]) for l in lines if l.strip())
    return pid, row.get("include", ""), hr, hw
ids = sorted(f[:-4] for f in os.listdir(p0.RAW_DIR))
with Pool(24) as pool:
    R = list(pool.imap_unordered(one, ids, chunksize=20))
json.dump(R, open("p0_probe.json", "w"))
C = collections.Counter()
for pid, inc, hr, hw in R:
    C["articles"] += 1
    C[f"include={inc}"] += 1
    C["focal term anywhere in text"] += hw > 0
    C["focal term inside analysed regions"] += hr > 0
    C["focal term only outside the regions"] += hw > 0 and hr == 0
print(json.dumps(C, indent=0))
