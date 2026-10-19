import glob, json, collections, re
from multiprocessing import Pool
def one(p):
    C = collections.Counter(); arts = set(); ex = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        for s in r["sentences"]:
            for u in s.get("units", []):
                C["units"] += 1
                curly = re.search(r"[A-Za-z]s’(?=\s|$|[,.;:])", u["text"])
                C["units with plural s’"] += bool(curly)
                bad = [a for a, w, i in u.get("atom2word", []) if w in ("’", "‘") and a.split("/")[1].startswith("P")]
                if bad:
                    C["units with a quote mark parsed as a verb"] += 1; arts.add(r["pmcid"])
                    C["... of them with plural s’"] += bool(curly)
                    if len(ex) < 3: ex.append(u["text"][:120])
    return C, arts, ex
T = collections.Counter(); A = set(); E = []
with Pool(25) as pool:
    for c, a, e in pool.imap_unordered(one, sorted(glob.glob("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"))):
        T.update(c); A |= a; E += e
print(dict(T), "articles", len(A)); print(E[:6])
