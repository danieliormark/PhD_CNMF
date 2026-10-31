# Soft model-name similarity in W: for pairs of child hyperedges identical except for one named model each,
# the idf-weighted Jaccard with the model pair counted as s of a shared word; share reaching tau 0.4 / 0.5.
import collections, itertools, math, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
hs = set()
for r in recs:
    for c in [c for p in r["parents"] for c in p["children"]] + r["cousins"]:
        k = key(c["atoms"], "tier")
        if k: hs.add(k)
H = len(hs); df = collections.Counter(a for k in hs for a in k)
idf = {a: math.log(H / (d + 1)) + 1 for a, d in df.items()}
isnamed = lambda a: a.endswith("/C/focal") and a.split("/")[0] not in GENERIC
FAMS = {level(a.split("/")[0], "family") for k in hs for a in k if isnamed(a)}
def lineage(x):   # domain-adapted / derived name containing another family name (mentalbert -> bert)
    return {f for f in FAMS if f != x and len(f) >= 3 and (x.endswith(f) or x.startswith(f))}
def rel(a, b):
    if level(a, "version") == level(b, "version"): return "same family, different tier"
    fa, fb = level(a, "family"), level(b, "family")
    if "chatgpt" in (a, b) and "gpt" in (fa, fb): return "chatgpt vs other gpt"
    if fa == fb: return "bare family vs versioned" if a == fa or b == fb else "same family, different version"
    if fb in lineage(fa) or fa in lineage(fb): return "domain-adapted vs base"
    return "different families"
grp = collections.defaultdict(set)
for k in hs:
    m = [a for a in k if isnamed(a)]
    if len(m) == 1: grp[k - {m[0]}].add(m[0])
S = {"same family, different tier": [0.95], "same family, different version": [0.8], "bare family vs versioned": [0.66],
     "chatgpt vs other gpt": [0.5, 0.66], "domain-adapted vs base": [0.5]}
res = collections.defaultdict(list); pure = collections.Counter()
for rest, ms in grp.items():
    R = sum(idf[a] for a in rest)
    for a, b in itertools.combinations(sorted(ms), 2):
        rl = rel(a.split("/")[0], b.split("/")[0])
        if rl not in S: continue
        if not rest: pure[rl] += 1
        for s in S[rl]:
            inter = R + s * min(idf[a], idf[b]); res[(rl, s)].append(inter / (R + idf[a] + R + idf[b] - inter) if False else inter / (2 * R + idf[a] + idf[b] - inter))
for (rl, s), v in sorted(res.items()):
    v.sort(); n = len(v)
    print(f"{rl:32s} s={s:4.2f}: pairs {n:5d} (model-only {pure[rl]:3d}) | median sim {v[n//2]:.2f} | reach 0.4 {sum(x>=0.4 for x in v)/n:.2f} | reach 0.5 {sum(x>=0.5 for x in v)/n:.2f}")
print("idf chatgpt", round(idf["chatgpt/C/focal"], 2), "gpt_4", round(idf["gpt_4/C/focal"], 2), "mentalbert", round(idf.get("mentalbert/C/focal", 0), 2))
print("domain-adapted names found:", sorted(f for f in FAMS if lineage(f))[:40])
