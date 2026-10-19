import glob, json, collections
from multiprocessing import Pool
SH = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/shards/"
def one(p):
    C = collections.Counter()
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); C["units"] += 1; C["parents"] += len(r["parents"])
        groups = collections.Counter(pp["content_key"] for pp in r["parents"])
        C["clauses (distinct parent word sets)"] += len(groups)
        C["units with a clause given 2+ parents"] += any(v > 1 for v in groups.values())
        C["extra parents from rotation"] += sum(v - 1 for v in groups.values())
        C["units with 2+ distinct clauses"] += len(groups) > 1
    return C
def rep(p):
    r = json.load(open(p)); c = r["counts"]
    return collections.Counter({k: v for k, v in c.items() if "focal_in_periphery" in k or k in ("cousins",)})
T = collections.Counter(); R = collections.Counter()
with Pool(25) as pool:
    for c in pool.imap_unordered(one, sorted(glob.glob(SH + "g3_test_*.jsonl"))): T.update(c)
    for c in pool.imap_unordered(rep, sorted(glob.glob(SH + "g3_report_*.json"))): R.update(c)
print(json.dumps(T, indent=0)); print(json.dumps(R, indent=0))
