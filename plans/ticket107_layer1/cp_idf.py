# child×parent: repetition of a (child, parent) pair (tf), parent frequency of children (df), and the share of the
# matrix's squared mass held by the most frequent children, binary vs idf. Children keyed as in the builder
# (tier-level names, numbers left out); parents keyed by their record content_key. Cousin ties at 1/sqrt(2).
import collections, math, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
tf = collections.Counter(); kind = {}
for r in recs:
    for p in r["parents"]:
        for c in p["children"]:
            k = key(c["atoms"], "tier")
            if k: tf[(k, p["content_key"])] += 1; kind[(k, p["content_key"])] = 1.0
    pk = {p["id"]: p["content_key"] for p in r["parents"]}
    for c in r["cousins"]:
        k = key(c["atoms"], "tier")
        for pid in c.get("parents") or []:
            if k and pid in pk and (k, pk[pid]) not in kind:
                tf[(k, pk[pid])] += 1; kind[(k, pk[pid])] = 0.5 ** 0.5
P = len({p for _, p in tf}); df = collections.Counter(k for k, _ in tf)
print("parents", P, "child hyperedges", len(df), "ties", len(tf))
c = collections.Counter(min(v, 3) for v in tf.values()); print("tf of a (child, parent) pair:", {k: c[k] for k in sorted(c)})
idf = {k: math.log(P / (d + 1)) + 1 for k, d in df.items()}
top = sorted(df, key=lambda k: -df[k])
print("most frequent children:", [(" ".join(sorted(a.split("/")[0] for a in k)), df[k], round(idf[k], 2)) for k in top[:8]])
cd = collections.Counter(1 if d == 1 else 2 if d == 2 else 5 if d <= 5 else 50 if d <= 50 else 51 for d in df.values())
print("children by number of parents:", {("1", "2", "3-5", "6-50", "51+")[i]: cd[b] for i, b in enumerate((1, 2, 5, 50, 51))})
for lab, w in [("binary", lambda k: 1.0), ("idf", lambda k: idf[k])]:
    m = collections.Counter()
    for (k, p), v in kind.items(): m[k] += (v * w(k)) ** 2
    tot = sum(m.values())
    print(f"{lab:6s}: top 20 children hold {sum(m[k] for k in top[:20]) / tot:.3f} of squared mass; children in 51+ parents {sum(m[k] for k in df if df[k] > 50) / tot:.3f}; in 1 parent {sum(m[k] for k in df if df[k] == 1) / tot:.3f}")
