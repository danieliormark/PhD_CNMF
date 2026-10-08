"""Q6: every 'LLM(s)' occurrence in the G2 v3 corpus, classified per article by the article's own definition and evidence."""
import json, re, collections, glob, sys
SH = sorted(glob.glob("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"))
LLM = re.compile(r'(?i)(?<![a-zA-Z0-9])LLMs?(?![a-zA-Z0-9])')
# definitions: "<expansion> (LLM)" or "LLM (<expansion>)" or "LLM, <expansion>" / "LLM = "
DEF_BEFORE = re.compile(r'((?:[\w\-‐’\']+[\s\-‐]+){1,6})\(\s*LLMs?\s*[),;]')
DEF_AFTER = re.compile(r'(?<![a-zA-Z0-9])LLMs?\s*\(\s*([^()]{3,60})\)')
LANGM = re.compile(r'(?i)language[\s\-]+models?')
OTHER_EVID = re.compile(r'(?i)(?<![a-zA-Z0-9])(chat[ -]?gpt|gpt[- ]?[345]|bert|gemini|claude|llama|deepseek|qwen|generative ai|chatbot|transformer|prompt)')
arts = {}
for f in SH:
    for l in open(f):
        a = json.loads(l)
        texts = [p.get("orig") or p.get("parsed") or "" for s in a["sentences"] for p in s.get("pieces", [])]
        full = " ".join(texts)
        occ = [(m.group(0), full[max(0, m.start()-90):m.end()+60]) for m in LLM.finditer(full)]
        if not occ:
            continue
        defs = [("before", m.group(1).strip()) for m in DEF_BEFORE.finditer(full)] + [("after", m.group(1).strip()) for m in DEF_AFTER.finditer(full)]
        arts[a["pmcid"]] = dict(n=len(occ), forms=collections.Counter(o[0] for o in occ), defs=defs,
                                langm=len(LANGM.findall(full)), other=len(OTHER_EVID.findall(full)), ex=occ[:3], shard=f[-9:-6])
json.dump(arts, open("llm_scan.json", "w"), indent=0)
print("articles in corpus scanned:", len(SH), "shards; articles with LLM:", len(arts), "occurrences:", sum(v["n"] for v in arts.values()))
