# article×article (approach 2): consequences of (a) the decided weight w(A,B) = Σ_p 1/(n_p−1) / |authors(A) ∪ authors(B)|
# and (b) symmetric rescaling w'(A,B) = w(A,B) / sqrt(s_A s_B), s = row Euclidean norm (influence of a row = Σ w²).
import collections, itertools, json, math, os
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_stats.py")).read().split("def q(")[0])
au = collections.defaultdict(set)
for r in B: au[r["pmcid"]].add(r["person"])
pa = collections.defaultdict(set)
for a, ps in au.items():
    for p in ps: pa[p].add(a)
w = collections.Counter()
for p, arts in pa.items():
    if len(arts) < 2: continue
    for x, y in itertools.combinations(sorted(arts), 2): w[(x, y)] += 1 / (len(arts) - 1)
for (x, y) in w: w[(x, y)] /= len(au[x] | au[y])
row = collections.defaultdict(dict)
for (x, y), v in w.items(): row[x][y] = v; row[y][x] = v
s = {a: math.sqrt(sum(v * v for v in d.values())) for a, d in row.items()}
wb = {e: v / math.sqrt(s[e[0]] * s[e[1]]) for e, v in w.items()}
rowb = collections.defaultdict(dict)
for (x, y), v in wb.items(): rowb[x][y] = v; rowb[y][x] = v
A = sorted(row); team = np.array([len(au[a]) for a in A]); deg = np.array([len(row[a]) for a in A])
ia = np.array([sum(v * v for v in row[a].values()) for a in A]); ib = np.array([sum(v * v for v in rowb[a].values()) for a in A])
from scipy.stats import spearmanr
print("tied articles", len(A), "ties", len(w))
for nm, i in (("(a)", ia), ("(b)", ib)):
    m = np.median(i)
    print(nm, "influence/median p10 %.2f p90 %.2f max %.1f" % (np.quantile(i, .1) / m, np.quantile(i, .9) / m, i.max() / m),
          "| Spearman with team size %.2f, with number of tied articles %.2f" % (spearmanr(i, team)[0], spearmanr(i, deg)[0]))
    for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 1000)):
        sel = (team >= lo) & (team <= hi); print("    team %d-%d: n=%d median influence/overall median %.2f" % (lo, hi, sel.sum(), np.median(i[sel]) / m))
# examples: strongest and weakest ties, before and after
ev = sorted(w, key=lambda e: w[e])
print("\nweakest ties (a) -> (b):")
for e in ev[:3] + ev[len(ev)//2:len(ev)//2+2] + ev[-3:]:
    x, y = e; print("  teams %d,%d shared %d, ties of each %d,%d : %.3f -> %.3f" % (len(au[x]), len(au[y]), len(au[x] & au[y]), len(row[x]), len(row[y]), w[e], wb[e]))
# isolated pairs: the two articles have only this one tie -> (b) sets it to 1 whatever its strength
iso = [e for e in w if len(row[e[0]]) == 1 and len(row[e[1]]) == 1]
print("\npairs tied only to each other:", len(iso), "; their (a) weights median %.3f min %.3f" % (np.median([w[e] for e in iso]), min(w[e] for e in iso)))
print("share of articles with exactly one tie:", round(float((deg == 1).mean()), 2))
# (c) partial rescaling, as for article rows: divide by (s_A s_B)^(1/4), so influence ∝ sqrt of the original
wc = {e: v / (s[e[0]] * s[e[1]]) ** 0.25 for e, v in w.items()}
ic = collections.Counter()
for (x, y), v in wc.items(): ic[x] += v * v; ic[y] += v * v
ic = np.array([ic[a] for a in A]); m = np.median(ic)
print("\n(c) influence/median p10 %.2f p90 %.2f max %.1f" % (np.quantile(ic, .1) / m, np.quantile(ic, .9) / m, ic.max() / m))
for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 1000)):
    sel = (team >= lo) & (team <= hi); print("    team %d-%d: median influence/overall median %.2f" % (lo, hi, np.median(ic[sel]) / m))
e = min(iso, key=lambda e: w[e]); print("weakest isolated pair: teams %d,%d: (a) %.3f (b) %.3f (c) %.3f" % (len(au[e[0]]), len(au[e[1]]), w[e], wb[e], wc[e]))
