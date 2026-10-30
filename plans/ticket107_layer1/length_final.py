import json, collections, numpy as np, scipy.stats as ss, statsmodels.api as sm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"w_stats.py")).read().split('print("articles"')[0])
E=json.load(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"length_extra.json")))
D="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/"
W=json.load(open(D+"openalex_authors/works_metrics.json")); Y=json.load(open(D+"author_coverage/publication_dates.json"))["years"]
hyp=collections.defaultdict(set); units=collections.Counter()
for r in recs:
    units[r["pmcid"]]+=1
    for c in [c for p in r["parents"] for c in p["children"]]+r["cousins"]: hyp[r["pmcid"]].add(c["content_key"])
def cat(tags):
    s=set(tags)
    if not s: return "unknown"
    if s&{"Systematic Review","Meta-Analysis","Network Meta-Analysis","Scoping Review","Evidence Synthesis"}: return "systematic review"
    if s&{"Randomized Controlled Trial","Clinical Trial","Clinical Trial Protocol","Clinical Trial, Phase I","Clinical Trial, Phase II","Pragmatic Clinical Trial","Equivalence Trial","Clinical Study"}: return "trial/protocol"
    if "Review" in s: return "narrative review"
    if s&{"Editorial","Letter","Comment","News","Interview","Introductory Journal Article","Guideline","Consensus Statement"}: return "editorial/letter/guideline"
    return "research article"
R=[]
for a in sorted(trial):
    w=W.get(E["ids"][a]["wid"] or "")
    yr=(Y.get(a) or {}).get("openalex") or (w and w["year"])
    R.append(dict(a=a,c=cat(E["types"][a]),h=len(hyp[a]),u=units[a],s=E["sent"][a],sub=E["inc_sub"][a],corp=E["inc_corp"][a],
        cited=w and w["cited"],pct=w and w["pct"],yr=int(yr) if yr else None,new=a not in Y))
print("types",collections.Counter(r["c"] for r in R).most_common())
K=[r for r in R if r["c"]!="unknown" and r["s"]>0]
h=np.array([r["h"] for r in K]); s=np.array([r["s"] for r in K]); u=np.array([r["u"] for r in K])
y=np.array([r["c"] in ("narrative review","systematic review") for r in K]).astype(float)
print("n",len(K),"review share",y.mean().round(3))
q=np.quantile(h,[.25,.5,.75,.9,.95]); edges=[0,*q,1e9]
print("\nband (distinct child hyperedges) | n | narrative rev | systematic rev | editorial | median sentences | median G3 units")
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(h>lo)&(h<=hi); sub=[r for r,k in zip(K,m) if k]; c=collections.Counter(r["c"] for r in sub)
    print(f"({lo:.0f},{hi:.0f}] n={len(sub)} {c['narrative review']/len(sub):.3f} {c['systematic review']/len(sub):.3f} {c['editorial/letter/guideline']/len(sub):.3f} {np.median(s[m]):.0f} {np.median(u[m]):.0f}")
def lo_(x,yv,name):
    m=sm.Logit(yv,sm.add_constant(x)).fit(disp=0); p=m.params[1:]; ci=m.conf_int()[1:]
    print(name,"OR per doubling",np.exp(p*np.log(2)).round(2),"CI",np.exp(ci*np.log(2)).round(2).tolist(),"p",m.pvalues[1:].round(5))
print("\nlogit P(review):")
lo_(np.log(h),y,"  log distinct child hyperedges:")
lo_(np.log(s),y,"  log total sentences:")
lo_(np.column_stack([np.log(s),np.log(u/s)]),y,"  joint [log sentences, log share of sentences with focal extraction]:")
yn=np.array([r["c"]=="narrative review" for r in K]).astype(float); ys=np.array([r["c"]=="systematic review" for r in K]).astype(float)
lo_(np.log(h),yn,"  narrative only ~ log hyperedges:"); lo_(np.log(h),ys,"  systematic only ~ log hyperedges:")
print("\nSpearman hyperedges vs sentences",ss.spearmanr(h,s)[0].round(3))
print("\ncitations (Spearman with distinct child hyperedges)")
for k,lab in [("sub","within subset"),("corp","within corpus"),("cited","total (OpenAlex)"),("pct","citation percentile (field/year-normalised)")]:
    vv=[(r["h"],r[k]) for r in R if r[k] is not None]; vr=[(r["h"],r[k]) for r in R if r[k] is not None and r["c"]=="research article"]
    rho=ss.spearmanr(*zip(*vv)); rr=ss.spearmanr(*zip(*vr))
    # within publication year
    by=collections.defaultdict(list)
    for r in R:
        if r[k] is not None and r["yr"]: by[r["yr"]].append((r["h"],r[k]))
    wy=[(len(v),ss.spearmanr(*zip(*v))[0]) for v in by.values() if len(v)>=30 and len(set(x[1] for x in v))>1]
    wavg=sum(n*x for n,x in wy)/sum(n for n,_ in wy)
    print(f"  {lab}: all rho={rho[0]:.3f} p={rho[1]:.2g} n={len(vv)} | research articles rho={rr[0]:.3f} | within-year weighted rho={wavg:.3f}")
print("\nmean incoming citations by band: within subset / within corpus / share cited at all in corpus ; research vs review")
for lo,hi in zip(edges[:-1],edges[1:]):
    out=[]
    for lab,f in [("research",lambda r:r["c"]=="research article"),("review",lambda r:r["c"] in ("narrative review","systematic review"))]:
        g=[r for r in R if f(r) and lo<r["h"]<=hi]
        out.append(f"{lab} n={len(g)} {np.mean([r['sub'] for r in g]):.2f}/{np.mean([r['corp'] for r in g]):.2f}/{np.mean([r['corp']>0 for r in g]):.2f}")
    print(f"({lo:.0f},{hi:.0f}]"," | ".join(out))
print("\nyears of trial articles",sorted(collections.Counter(r["yr"] for r in R if r["yr"]).items()),"no year",sum(1 for r in R if not r["yr"]))
