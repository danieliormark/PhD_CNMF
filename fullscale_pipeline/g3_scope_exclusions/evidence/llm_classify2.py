import json, re, collections
A = json.load(open("llm_scan.json"))
SKIP = {"of", "and", "the", "a", "an", "for", "in", "on", "with", "to"}
def acronym(exp):
    """the shortest tail of exp whose word initials spell 'llm' (stop words skipped), else None"""
    ws = [w for w in re.split(r"[\s\-‐/]+", exp) if w]
    for k in range(2, min(6, len(ws)) + 1):
        tail = ws[-k:]
        ini = "".join(w[0].lower() for w in tail if w.lower() not in SKIP)
        if ini == "llm" or ini == "llms":
            return " ".join(tail)
    return None
LM = re.compile(r"(?i)language")
res = {}
for k, v in A.items():
    exps = [acronym(d) for t, d in v["defs"] if t == "before"]
    exps = [e for e in exps if e]
    lm = [e for e in exps if LM.search(e)]
    oth = [e for e in exps if not LM.search(e)]
    if lm and oth: c = "1 both"
    elif lm: c = "2 defined: language model"
    elif oth: c = "3 defined: other"
    elif v["langm"] > 0: c = "4 undefined, 'language model' in article"
    elif v["other"] > 0: c = "5 undefined, other LLM evidence"
    else: c = "6 undefined, no evidence"
    res[k] = dict(c=c, oth=oth, lm=lm)
C = collections.Counter(r["c"] for r in res.values()); O = collections.Counter()
for k, r in res.items(): O[r["c"]] += A[k]["n"]
for c in sorted(C): print(f"{c:45} {C[c]:7} {O[c]:9}")
print("total", len(A), sum(O.values()))
print("\nother expansions:", collections.Counter(e.lower() for r in res.values() for e in r["oth"]).most_common(60))
json.dump(res, open("llm_classes2.json", "w"))
