import json, re, collections, sys
sys.argv = ["x", sys.argv[1], sys.argv[2]]
src = open(__file__.replace("new_articles", "rule")).read().split('if __name__ == "__main__":')[0]
exec(src)
c = open("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/focal_gap_classify.py").read()
exec(c[c.index("FAM ="):c.index("S = collections")])
F = json.load(open(sys.argv[1])); R = json.load(open(sys.argv[2]))
first = {f["form"]: is_model(f["form"]) for f in F}
NOISE_A = re.compile(r"(?i)^(palm|bard)s?$|^palm")       # group-1 forms that are mostly not models (the hand, a database)
cand = {}
for a, r in R.items():
    if r["status"] == "ok": continue
    hits = collections.Counter()
    for o in r["occ"]:
        t = o["tok"]; k = cls(t)
        if k[0] == "A" and not NOISE_A.match(t) and not re.fullmatch(r"GPT|Gemini|Bard|Llama|Mistral|PaLM|Gemma", t): hits[("1-3", t)] += 1
        elif first.get(t) == "model": hits[("4", t)] += 1
        elif first.get(t) == "tool": hits[("tool", t)] += 1
    if any(g in ("1-3", "4") for g, _ in hits): cand[a] = hits
print("candidate new articles", len(cand))
print("status of candidates", collections.Counter(R[a]["status"] for a in cand))
by = collections.Counter("1-3 and 4" if {"1-3", "4"} <= {g for g, _ in h} else ("1-3 only" if any(g == "1-3" for g, _ in h) else "4 only") for h in cand.values())
print(by)
for a in list(cand)[:15]: print(a, R[a]["status"], dict(cand[a].most_common(5)))
json.dump({a: [[g, t, n] for (g, t), n in h.items()] for a, h in cand.items()}, open("new_article_candidates.json", "w"))
