import collections, glob, json, statistics as st
CS="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/"
PG="/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
trial={l.strip() for l in open(CS+"psychology_trial_pmcids.txt") if l.strip()}
recs,dem=[],set()
for f in glob.glob(PG+"g3_v2_lm_demotion/demoted/g3_test_*.jsonl"):
    for l in open(f,encoding="utf-8"):
        r=json.loads(l)
        if r["pmcid"] in trial: dem.add(r["pmcid"]); recs.append(r)
for l in open(PG+"g3_v2/g3_test_v2.jsonl",encoding="utf-8"):
    p=l[11:l.index('"',11)]
    if p in trial and p not in dem: recs.append(json.loads(l))
print("articles",len(recs))
r0=recs[0]; print("parent keys",list(r0["parents"][0].keys())[:15]); print("child keys",list(r0["parents"][0]["children"][0].keys())[:15])
tf=collections.Counter(); arts=collections.defaultdict(set); par=collections.defaultdict(set); ptf=collections.Counter(); parts=collections.defaultdict(set)
childper=[]; parper=[]
for r in recs:
    a=r["pmcid"]; cs=set(); ps=set()
    for p in r["parents"]:
        pk=p.get("content_key"); ps.add(pk); ptf[(pk,a)]+=1; parts[pk].add(a)
        for c in p["children"]:
            k=c["content_key"]; tf[(k,a)]+=1; arts[k].add(a); par[k].add(pk); cs.add(k)
    for c in r["cousins"]:
        k=c["content_key"]; tf[(k,a)]+=1; arts[k].add(a); cs.add(k)
    childper.append(len(cs)); parper.append(len(ps))
def dist(name,vals,cuts):
    n=len(vals); c=collections.Counter(min(v,cuts[-1]) for v in vals)
    print(name,"n",n,{(f"{k}+" if k==cuts[-1] else k):round(100*c[k]/n,1) for k in cuts}, "max",max(vals))
dist("child: articles per hyperedge",[len(v) for v in arts.values()],[1,2,3,4,5,10])
dist("child: tf per (hyperedge,article)",list(tf.values()),[1,2,3,5])
dist("child: parents per hyperedge (children only)",[len(par[k]) for k in arts if par[k]],[1,2,3,5,10])
print("child hyperedges with no parent (cousin only):",sum(1 for k in arts if not par[k]))
dist("parent: articles per parent",[len(v) for v in parts.values()],[1,2,3])
def q(v): v=sorted(v); n=len(v); return [v[int(n*x)] for x in (0.1,0.5,0.9)]+[v[-1]]
print("child hyperedges per article p10/p50/p90/max",q(childper)); print("parents per article p10/p50/p90/max",q(parper))
top=sorted(arts,key=lambda k:-len(arts[k]))[:12]
print("most frequent:",[(k[:40],len(arts[k])) for k in top])
print("distinct pmcids",len({r["pmcid"] for r in recs}))
ca=collections.defaultdict(set); pa=collections.defaultdict(set); occ=collections.Counter(); words={}
for r in recs:
    for p in r["parents"]:
        pa[r["pmcid"]].add(p["content_key"])
        for c in p["children"]: ca[r["pmcid"]].add(c["content_key"]); occ[r["pmcid"]]+=1; words[c["content_key"]]=" ".join(sorted(x["atom"] for x in c["atoms"]))
    for c in r["cousins"]: ca[r["pmcid"]].add(c["content_key"]); occ[r["pmcid"]]+=1; words[c["content_key"]]=" ".join(sorted(x["atom"] for x in c["atoms"]))
print("distinct child hyperedges per article p10/p50/p90/max",q([len(v) for v in ca.values()]))
print("child occurrences per article p10/p50/p90/max",q(list(occ.values())))
print("parents per article p10/p50/p90/max",q([len(v) for v in pa.values()]))
print("most frequent:",[(words[k],len(arts[k])) for k in top])
