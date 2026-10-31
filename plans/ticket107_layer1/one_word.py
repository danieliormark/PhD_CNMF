# One-word child hyperedges in child×parent, and parent frequency vs article frequency of child hyperedges.
import collections, math, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
par = collections.defaultdict(set); art = collections.defaultdict(set)
for r in recs:
    pk = {p["id"]: p["content_key"] for p in r["parents"]}
    for p in r["parents"]:
        for c in p["children"]:
            k = key(c["atoms"], "tier")
            if k: par[k].add(p["content_key"]); art[k].add(r["pmcid"])
    for c in r["cousins"]:
        k = key(c["atoms"], "tier")
        if not k: continue
        art[k].add(r["pmcid"])
        for pid in c.get("parents") or []:
            if pid in pk: par[k].add(pk[pid])
C = [k for k in par]; P = len({p for v in par.values() for p in v}); N = len({r["pmcid"] for r in recs})
one = [k for k in C if len(k) == 1]
ties = sum(len(par[k]) for k in C); t1 = sum(len(par[k]) for k in one)
print(f"children in child×parent {len(C)}, one-word {len(one)} ({len(one)/len(C):.3f}); their share of ties {t1/ties:.3f}")
band = lambda d: "1" if d == 1 else "2-5" if d <= 5 else "6-50" if d <= 50 else "51+"
for b in ("1", "2-5", "6-50", "51+"):
    ks = [k for k in C if band(len(par[k])) == b]
    print(f"  children in {b:4s} parents: {len(ks):6d}, one-word share {sum(len(k)==1 for k in ks)/len(ks):.3f}")
print("examples of one-word children by parent frequency:")
for lo, hi in [(1, 1), (2, 5), (6, 50), (51, 10**6)]:
    ks = sorted([k for k in one if lo <= len(par[k]) <= hi], key=lambda k: -len(par[k]))
    print(f"  {lo}-{hi}: {len(ks)}", [next(iter(k)).split('/')[0] for k in ks[:12]])
import scipy.stats as ss
print("Spearman parent frequency vs article frequency", round(ss.spearmanr([len(par[k]) for k in C], [len(art[k]) for k in C])[0], 3))
ratio = sorted(C, key=lambda k: -len(par[k]) / len(art[k]))
print("children used in many claims of few articles:", [(" ".join(sorted(a.split('/')[0] for a in k)), len(par[k]), len(art[k])) for k in ratio[:8]])
ip = lambda k: math.log(P / (len(par[k]) + 1)) + 1; ia = lambda k: math.log(N / (len(art[k]) + 1)) + 1
rp = {k: i for i, k in enumerate(sorted(C, key=ip))}; ra = {k: i for i, k in enumerate(sorted(C, key=ia))}
top_p = set(sorted(C, key=ip)[:500]); top_a = set(sorted(C, key=ia)[:500])
print("overlap of the 500 most down-weighted children under the two counts", len(top_p & top_a))
