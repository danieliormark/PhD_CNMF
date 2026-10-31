# child×parent: parent-frequency idf vs article-frequency idf. Which children change weight, and how much of the
# matrix's squared mass moves. Ties binary (cousin ties not separated here: the comparison is between idf bases).
import collections, math, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "one_word.py")).read().split("C = [k for k in par]")[0])
C = list(par); P = len({p for v in par.values() for p in v}); N = len({r["pmcid"] for r in recs})
ip = {k: math.log(P / (len(par[k]) + 1)) + 1 for k in C}; ia = {k: math.log(N / (len(art[k]) + 1)) + 1 for k in C}
def share(w):
    m = {k: len(par[k]) * w[k] ** 2 for k in C}; t = sum(m.values()); return {k: v / t for k, v in m.items()}
sp, sa = share(ip), share(ia)
rep = [k for k in C if len(par[k]) >= 5 and len(par[k]) / len(art[k]) >= 3]
print(f"children with 5+ claims and 3+ claims per article: {len(rep)}; share of matrix mass: parent idf {sum(sp[k] for k in rep):.3f}, article idf {sum(sa[k] for k in rep):.3f}")
moved = sum(abs(sp[k] - sa[k]) for k in C) / 2
print(f"share of the matrix's squared mass that moves between children when switching basis: {moved:.3f}")
gen = [k for k in C if len(art[k]) > 100]
print(f"children in 100+ articles: {len(gen)}; share: parent idf {sum(sp[k] for k in gen):.3f}, article idf {sum(sa[k] for k in gen):.3f}")
