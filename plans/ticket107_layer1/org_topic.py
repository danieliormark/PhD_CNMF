import csv,collections,json
FP="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"; CS=FP+"corpus_statistics/"
T={l.strip() for l in open(CS+"psychology_trial_pmcids.txt") if l.strip()}
meta={r["pmcid"]:r for r in csv.DictReader(open(CS+"subset_candidates_by_article.csv")) if r["pmcid"] in T}
org={r["institution"]:r["organisation"] for r in csv.DictReader(open(CS+"institution_organisations_psychology_trial.csv"))}
oa=collections.defaultdict(set); pa=collections.defaultdict(set)
for l in open(FP+"diagnostics/author_coverage/author_table_v2.jsonl"):
    r=json.loads(l)
    if r["status"]=="kept" and r["pmcid"] in T:
        pa[r["person"]].add(r["pmcid"])
        for i in r["institutions"]: oa[org[i.rsplit('/',1)[-1]]].add(r["pmcid"])
def agg(groups,k="topic"):
    num=den=0
    for g in groups:
        g=list(g); n=len(g)
        if n<2: continue
        c=collections.Counter(meta[a][k] for a in g); num+=sum(v*(v-1) for v in c.values()); den+=n*(n-1)
    return round(num/den,3), den//2
for name,G in [("organisations",oa),("persons",pa)]:
    for lo,hi in [(2,5),(6,15),(16,1000)]:
        gs=[g for g in G.values() if lo<=len(g)<=hi]; t,pairs=agg(gs); s,_=agg(gs,"subfield")
        print(f"{name} with {lo}-{hi} trial articles: n={len(gs)} pairs={pairs} same topic {t} same subfield {s}")
