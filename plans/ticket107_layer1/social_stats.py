import collections, csv, json, math, numpy as np
FP="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"; CS=FP+"corpus_statistics/"
trial={l.strip() for l in open(CS+"psychology_trial_pmcids.txt") if l.strip()}
org={r["institution"]:r["organisation"] for r in csv.DictReader(open(CS+"institution_organisations_psychology_trial.csv"))}
B=[]
for l in open(FP+"diagnostics/author_coverage/author_table_v2.jsonl"):
    r=json.loads(l)
    if r["status"]=="kept" and r["pmcid"] in trial: B.append(r)
def q(v,ps=(.1,.25,.5,.75,.9,.95,.99)): v=np.array(v); return [round(float(np.quantile(v,p)),1) for p in ps]+[int(v.max())]
team=collections.Counter(r["pmcid"] for r in B)
ts=list(team.values()); print("bylines",len(B),"articles",len(team))
print("team size p10..p99,max",q(ts)); c=collections.Counter(min(t,11) for t in ts); print("team size dist",{k:round(c[k]/len(ts),3) for k in sorted(c)})
pa=collections.defaultdict(set)
for r in B: pa[r["person"]].add(r["pmcid"])
np_=[len(v) for v in pa.values()]; c=collections.Counter(min(n,6) for n in np_); print("persons",len(pa),"articles per person",{k:c[k] for k in sorted(c)},"max",max(np_))
# author x article influence per article under schemes (squared-error influence = sum w^2)
print("\nper-article influence sum w^2 by team size: log-damped (current) | fractional 1/n | binary | 1/sqrt(n)")
for n in (1,2,3,5,10,20,50):
    ld=n*(math.log2(1+n)/n)**2; print(f" n={n}: {ld:.2f} | {1/n:.3f} | {n} | 1")
# affiliations
ninst=[len(r["institutions"]) for r in B]; c=collections.Counter(min(n,4) for n in ninst); print("\ninstitutions per byline",{k:round(c[k]/len(B),3) for k in sorted(c)})
missing=[i for r in B for i in r["institutions"] if i.rsplit('/',1)[-1] not in org]; print("institutions not in org map",len(missing))
O=lambda r:{org.get(i.rsplit('/',1)[-1],i.rsplit('/',1)[-1]) for i in r["institutions"]}
ao=collections.defaultdict(collections.Counter)   # article -> org -> bylines
for r in B:
    for o in O(r): ao[r["pmcid"]][o]+=1
no=[len(ao[a]) for a in team]; print("organisations per article",q(no),"articles with none",sum(1 for a in team if not ao[a]))
mx=[max(ao[a].values()) for a in team if ao[a]]; print("max bylines of one organisation on an article",q(mx))
share1=[max(ao[a].values())/team[a] for a in team if ao[a]]; print("share of team from top organisation, median",np.median(share1).round(2))
oa=collections.Counter(o for a in ao for o in ao[a]); print("articles per organisation",q(list(oa.values())))
# author x affiliation: per person, distinct orgs across bylines, and per-person tie counts
po=collections.defaultdict(collections.Counter); pn=collections.Counter()
for r in B:
    pn[r["person"]]+=1
    for o in O(r): po[r["person"]][o]+=1
nop=[len(po[p]) for p in pa]; c=collections.Counter(min(n,4) for n in nop); print("organisations per person",{k:c[k] for k in sorted(c)})
multi=[p for p in pa if len(pa[p])>=2]; nopm=[len(po[p]) for p in multi]; c=collections.Counter(min(n,4) for n in nopm); print(" among persons with 2+ articles",{k:c[k] for k in sorted(c)})
# journals
J=collections.Counter(r["journal"] for r in csv.DictReader(open(CS+"subset_candidates_by_article.csv")) if r["pmcid"] in trial) if "journal" in next(csv.DictReader(open(CS+"subset_candidates_by_article.csv"))) else None
print("\nsubset_candidates columns",list(next(csv.DictReader(open(CS+"subset_candidates_by_article.csv"))).keys()))
if J: print("journals",len(J),"articles per journal",q(list(J.values())),"top",J.most_common(5))
