# Layer 1 review checks. (1) child×article: an article's influence after the decided weights (log(1+tf)·idf, then
# row / ||row||^1/2, so influence = ||row||) against its number of distinct child hyperedges. (2) child×parent: a parent
# hyperedge's influence (sum of squared cell values, decided rule) and its number of ties.
import collections, math, os
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
tf = collections.Counter(); arts = collections.defaultdict(set)
nest = collections.Counter(); cous = collections.Counter()
for r in recs:
    pk = {p["id"]: p["content_key"] for p in r["parents"]}
    for p in r["parents"]:
        for c in p["children"]:
            k = key(c["atoms"], "tier")
            if k: tf[(k, r["pmcid"])] += 1; arts[k].add(r["pmcid"]); nest[(k, p["content_key"])] += 1
    for c in r["cousins"]:
        k = key(c["atoms"], "tier")
        if not k: continue
        tf[(k, r["pmcid"])] += 1; arts[k].add(r["pmcid"])
        for pid in c.get("parents") or []:
            if pid in pk: cous[(k, pk[pid])] += 1
N = len({a for _, a in tf}); idf_a = {k: math.log(N / (len(v) + 1)) + 1 for k, v in arts.items()}
sq = collections.Counter(); cnt = collections.Counter()
for (k, a), t in tf.items(): sq[a] += (math.log1p(t) * idf_a[k]) ** 2; cnt[a] += 1
A = sorted(sq); infl = np.array([math.sqrt(sq[a]) for a in A]); L = np.array([cnt[a] for a in A])
def rel(v): m = np.median(v); return round(float(np.quantile(v, .9) / m), 1), round(float(v.max() / m), 1)
print("child×article, ratio to the median article (p90, max): by hyperedge count", rel(L), "| intended sqrt(count)", rel(np.sqrt(L)),
      "| actual influence after weights", rel(infl))
print("  Spearman influence vs count", round(__import__("scipy.stats").stats.spearmanr(infl, L)[0], 3))
pairs = set(nest) | set(cous); P = len({p for _, p in pairs}); df = collections.Counter(k for k, _ in pairs)
idf_p = {k: math.log(P / (d + 1)) + 1 for k, d in df.items()}
pin = collections.Counter(); pt = collections.Counter()
for kp in pairs:
    n = nest[kp] + 0.633 * cous[kp]; pin[kp[1]] += (idf_p[kp[0]] * math.log1p(n)) ** 2; pt[kp[1]] += 1
v = np.array(list(pin.values())); t = np.array(list(pt.values()))
q = lambda x: [round(float(np.quantile(x, p)), 1) for p in (.1, .5, .9)] + [round(float(x.max()), 1)]
print("child×parent: ties per parent hyperedge p10/p50/p90/max", q(t), "| influence p10/p50/p90/max", q(v), "| max/median", round(float(v.max() / np.median(v)), 1))
