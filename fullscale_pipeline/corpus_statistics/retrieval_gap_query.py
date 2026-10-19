# Ticket 107 P1, D2 (owner 2026-10-19): which PMC open-access articles name a missed model in title/abstract but are not in
# R1's target list? Read-only Entrez esearch with R1's own filters (query_pmc_entrez.py: "open access"[filter], PubDate
# 2015-01 .. 2026-09, the R1 run of 2026-09-23 covered whole months up to September). Three term sets, each [Title/Abstract]:
#   GEN    the bare names added to focal_terms.py on 2026-09-26, after R1 ran (never used for retrieval)
#   ALIAS  spellings of listed terms found missing by the scan (D2 groups 1-3: CT-BERT, ChatGPT4, GPTs, ...)
#   G4     unlisted model names accepted by the draft group-4 rule (focal_group4_rule.py)
# PMC search ignores letter case, so a form whose lower case is also a name or word seen as noise in the scan (Robert,
# Albert, Herbert, ...) is left out of the query; everything returned is filtered later by the case-sensitive matcher.
#   python3 retrieval_gap_query.py SCAN.json FORMS.json OUT.json [FIELD] [SETS]      (needs NCBI_API_KEY for 10 requests/s)
# FIELD defaults to Title/Abstract; "Body - All Words" is what R1's [Text Word] becomes in PMC (full body text, without
# the abstract). SETS (comma list, default GEN,ALIAS,G4) may add OLD: R1's own [Title/Abstract] names, searched in FIELD.
import calendar, collections, json, os, re, sys, time, urllib.parse, urllib.request

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
API_KEY = os.environ.get("NCBI_API_KEY", "")
INTERVAL = 0.11 if API_KEY else 0.35
URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
GEN = ["BERT", "RoBERTa", "GPT", "Gemini", "Bard", "Llama", "Mistral", "PaLM", "Gemma", "language model", "language models"]
SUFFIX = re.compile(r"[-_](?:based|derived|driven|like|style|powered|assisted|generated|only|enhanced)$", re.I)


def esearch(term, mindate, maxdate):
    data = urllib.parse.urlencode(dict(db="pmc", term=term, retmax=10000, retmode="json", tool="csf3_tensor_pipeline",
                                       mindate=mindate, maxdate=maxdate, datetype="pdat", **({"api_key": API_KEY} if API_KEY else {}))).encode()
    for attempt in range(4):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=data, method="POST"), timeout=120).read())
            time.sleep(INTERVAL)
            return int(r["esearchresult"]["count"]), r["esearchresult"].get("idlist", [])
        except Exception as e:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"esearch failed: {term[:80]} {mindate}")


FIELD = sys.argv[4] if len(sys.argv) > 4 else "Title/Abstract"


def ids_for(terms):
    q = "(" + " OR ".join(f'"{t}"[{FIELD}]' for t in terms) + ') AND "open access"[filter]'
    out = set()
    for y in range(2015, 2027):
        n, ids = esearch(q, f"{y}/01/01", f"{y}/12/31" if y < 2026 else "2026/09/30")
        if n > 9999:                                      # split by month, as R1 did
            for m in range(1, 13 if y < 2026 else 10):
                n2, ids2 = esearch(q, f"{y}/{m:02d}/01", f"{y}/{m:02d}/{calendar.monthrange(y, m)[1]}")
                if n2 > 9999:                             # split the month in thirds
                    last = calendar.monthrange(y, m)[1]
                    for a, b in ((1, 10), (11, 20), (21, last)):
                        n3, ids3 = esearch(q, f"{y}/{m:02d}/{a:02d}", f"{y}/{m:02d}/{b:02d}")
                        assert n3 <= 9999, (terms[:3], y, m, a, n3)
                        out |= set(ids3)
                else:
                    out |= set(ids2)
        else:
            out |= set(ids)
    return {f"PMC{i}" for i in out}


if __name__ == "__main__":
    scan, forms_p, out_p = sys.argv[1:4]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rule_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "focal_group4_rule.py")).read().split('if __name__')[0]
    ns = {}; exec(rule_src, ns)
    cls_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "focal_gap_classify.py")).read()
    exec(cls_src[cls_src.index("FAM ="):cls_src.index("S = collections")], ns)
    R = json.load(open(scan)); F = json.load(open(forms_p))
    noise_lower = {f["form"].lower() for f in F if ns["is_model"](f["form"]) is None}
    def clean(t):
        t = SUFFIX.sub("", t).strip(".")
        t = re.sub(r"(?<=[A-Za-z])\d{2}$", "", t) if re.search(r"[A-Za-z]{3,}\d{2}$", t) else t   # glued citation digits
        return t if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_\- ]{2,40}", t) and t.lower() not in noise_lower else None
    alias = collections.Counter()
    for a, r in R.items():
        for o in r["occ"]:
            c = ns["cls"](o["tok"])
            if c[:2] in ("A1", "A2", "A3", "A5", "A6"):
                alias[o["tok"]] += 1
    g4 = [f["form"] for f in F if ns["is_model"](f["form"]) == "model"]
    want = (sys.argv[5] if len(sys.argv) > 5 else "GEN,ALIAS,G4").split(",")
    r1 = open("/mnt/hum01-rds/Basov/p91688di/query_pmc_entrez.py").read()
    OLD = re.findall(r'"([^"]+)"\[Title/Abstract\]', r1)
    sets = {"GEN": GEN, "OLD": OLD}
    for name, src in (("ALIAS", list(alias)), ("G4", g4)):
        seen, keep = set(), []
        for t in src:
            c = clean(t)
            if c and c.lower() not in seen:
                seen.add(c.lower()); keep.append(c)
        sets[name] = keep
    sets = {k: v for k, v in sets.items() if k in want}
    target = set(open(LCS + "target_pmcids.txt").read().split())
    print("field", FIELD, "sets", {k: len(v) for k, v in sets.items()}, flush=True)
    res = {"field": FIELD, "terms": sets, "ids": {}}
    for name, terms in sets.items():
        got = set()
        for i in range(0, len(terms), 80):
            got |= ids_for(terms[i:i + 80])
            print(f"{name}: {min(i + 80, len(terms))}/{len(terms)} terms, {len(got):,} articles, {len(got - target):,} not in R1", flush=True)
        res["ids"][name] = sorted(got)
    allnew = set().union(*map(set, res["ids"].values())) - target
    res["summary"] = {k: dict(found=len(v), not_in_r1=len(set(v) - target)) for k, v in res["ids"].items()}
    res["summary"]["all"] = dict(not_in_r1=len(allnew))
    json.dump(res, open(out_p, "w"), indent=0)
    print(json.dumps(res["summary"], indent=0))
