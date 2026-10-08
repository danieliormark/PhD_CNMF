"""Q5: orig vs fix -- fusions gained/lost (distinct lemma per unit) with text and verb-particle token distance."""
import json, re, collections
P = {'out', 'up', 'down', 'in', 'on', 'off', 'over'}
def fusions(pre, s):
    out = {}
    for l in open(f"{pre}_{s}/g3_test_{s:03d}.jsonl"):
        r = json.loads(l)
        kids = [k for p in r["parents"] for k in p["children"]] + r["cousins"]
        for k in kids:
            for a in k.get("atoms", []):
                m = re.match(r'([a-z]+(?:_[a-z]+)*)_(out|up|down|in|on|off|over)/P', a["atom"])
                if m:
                    ps = [pos for _, w, pos in a["src"] if w.lower() in P]
                    vs = [pos for _, w, pos in a["src"] if w.lower() not in P]
                    d = min((pp - vv for pp in ps for vv in vs), key=abs) if ps and vs else None
                    out[(r["uid"], a["atom"].split("/")[0])] = (d, r["text"])
    return out
gain, lost, both = {}, {}, {}
for s in [0, 20, 30, 40]:
    o, f = fusions("orig", s), fusions("fix", s)
    for k in f.keys() - o.keys(): gain[k] = f[k]
    for k in o.keys() - f.keys(): lost[k] = o[k]
    for k in f.keys() & o.keys(): both[k] = f[k]
print("fusions kept:", len(both), "gained:", len(gain), "lost:", len(lost))
print("distance, kept :", sorted(collections.Counter(v[0] for v in both.values()).items(), key=lambda t: (t[0] is None, t[0] or 0)))
print("distance, gained:", sorted(collections.Counter(v[0] for v in gain.values()).items(), key=lambda t: (t[0] is None, t[0] or 0)))
print("gained lemmas:", collections.Counter(k[1] for k in gain).most_common(40))
json.dump(dict(gain=[[k[0], k[1], v[0], v[1]] for k, v in gain.items()], lost=[[k[0], k[1], v[0], v[1]] for k, v in lost.items()],
               kept=[[k[0], k[1], v[0], v[1]] for k, v in both.items()]), open("q5_compare.json", "w"), indent=0)
print("=== lost")
for k, v in lost.items(): print(f"  {k[1]} d={v[0]} | {v[1][:130]}")
