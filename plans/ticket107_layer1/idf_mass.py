import collections, math
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"w_stats.py")).read().split('print("articles"')[0])
tf=collections.Counter(); arts=collections.defaultdict(set)
for r in recs:
    for c in [c for p in r["parents"] for c in p["children"]]+r["cousins"]:
        tf[(c["content_key"],r["pmcid"])]+=1; arts[c["content_key"]].add(r["pmcid"])
N=len({r["pmcid"] for r in recs})
idf=lambda df: math.log(N/(df+1))+1
band=lambda d: "1" if d==1 else "2" if d==2 else "3-5" if d<=5 else "6-50" if d<=50 else "51+"
sq=collections.Counter(); lin=collections.Counter(); n=collections.Counter()
for (k,a),t in tf.items():
    d=len(arts[k]); w=math.log1p(t)*idf(d); b=band(d); sq[b]+=w*w; lin[b]+=w; n[b]+=1
S,L,C=sum(sq.values()),sum(lin.values()),sum(n.values())
print("N",N,"idf(1)",round(idf(1),2),"idf(2)",round(idf(2),2),"idf(1143)",round(idf(1143),2))
for b in ["1","2","3-5","6-50","51+"]: print(f"df {b}: ties {n[b]/C:.3f} | share of sum w (KL-like) {lin[b]/L:.3f} | share of sum w^2 (squared error) {sq[b]/S:.3f}")
