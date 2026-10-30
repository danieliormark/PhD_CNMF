# Fill gaps for the 708 newer trial articles: types (MEDLINE tags via gap cache), sentences
# (same naive split, from the pure-text files for all 2,139 so both groups are counted alike),
# within-subset / within-corpus incoming citations from citation_dictionary_v2 reference texts.
import json, os, re, sys, collections
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics")
from fullscale_analysis import SENT_SPLIT
CS="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/"
D="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/"
L="/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
T=sorted({l.strip() for l in open(CS+"psychology_trial_pmcids.txt") if l.strip()})
V=json.load(open(D+"openalex_verified.json")); V.update({k:v for k,v in json.load(open(D+"openalex_verified_new.json")).items() if k not in V})
ids={}
for a in T:
    w=(V.get(a) or [None,None])[1] or {}
    ids[a]=dict(pmid=w.get("pmid"),doi=(w.get("doi") or "").lower() or None,wid=(w.get("id") or "").rsplit("/",1)[-1] or None)
PT=json.load(open(D+"pubmed_pubtypes.json")); PT.update(json.load(open(L+"gap_pubtypes_cache.json")))
types={a:[t["name"] for t in PT.get(ids[a]["pmid"] or "",[])] for a in T}
sent={}
for a in T:
    fs=[d+f for d in (L+"pure_text_corpus/",L+"pure_text_corpus_gap/") for f in [a+".txt"] if os.path.exists(d+f)]
    t=open(fs[0],encoding="utf-8",errors="replace").read().strip() if fs else ""
    sent[a]=len(SENT_SPLIT.split(t)) if t else 0
key={}
for a in T:
    key["pmc:"+a.lower()]=a
    if ids[a]["pmid"]: key["pmid:"+ids[a]["pmid"]]=a
    if ids[a]["doi"]: key["doi:"+ids[a]["doi"].replace("https://doi.org/","")]=a
trialset=set(T); inc_sub=collections.defaultdict(set); inc_corp=collections.defaultdict(set); citing=0
DOI=re.compile(r"10\.\d{4,9}/[^\s\"<>]+"); PMC=re.compile(r"PMC\d+"); NUM=re.compile(r"\b\d{7,8}\b")
for l in open(L+"citation_dictionary_v2.jsonl",encoding="utf-8"):
    r=json.loads(l); src=r["pmcid"]; citing+=1
    for c in r["cited"]:
        tx=c.get("reference_text") or ""; hits=set()
        for m in PMC.findall(tx): hits.add(key.get("pmc:"+m.lower()))
        for m in DOI.findall(tx): hits.add(key.get("doi:"+m.lower().rstrip(".,;)")))
        if c.get("doi"): hits.add(key.get("doi:"+c["doi"].lower()))
        for m in NUM.findall(tx): hits.add(key.get("pmid:"+m))
        for h in hits:
            if h and h!=src:
                inc_corp[h].add(src)
                if src in trialset: inc_sub[h].add(src)
json.dump(dict(types=types,sent=sent,inc_sub={a:len(inc_sub[a]) for a in T},inc_corp={a:len(inc_corp[a]) for a in T},ids=ids,citing_articles=citing),open(__import__("os").path.join(__import__("os").path.dirname(__file__),"length_extra.json"),"w"))
print("citing articles in dictionary",citing,"trial with types",sum(bool(v) for v in types.values()),"with sentences",sum(v>0 for v in sent.values()))
print("trial articles cited within subset",sum(1 for a in T if inc_sub[a]),"within corpus",sum(1 for a in T if inc_corp[a]))
