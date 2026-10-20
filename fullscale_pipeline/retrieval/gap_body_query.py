"""R1b candidates, full body text (owner 2026-10-20: search the whole text, to miss no true positive), ticket 107 P1.
PMC esearch with R1's filters ("open access"[filter], publication dates 2015-01-01..2026-09-30), each term searched in
[Body - All Words] (the body text) and in [Title/Abstract] (the body field leaves the abstract out). Terms:
  - R1's names that count as focal terms on their own (focal_terms.py PLAIN), R1's three body-text phrases, and BERT,
    RoBERTa, "language model(s)" (GENERIC names that need no article evidence);
  - the spellings of listed terms (D2 groups 1-3) and the unlisted model names (group 4) found by the corpus scan.
Left out, with the reason: GUARDED names (Claude, RITA, GROVER, KeAP, GPN, ...) and the GENERIC names that need article
evidence (GPT, Gemini, Bard, Llama, Mistral, PaLM, Gemma): the matcher counts them only in an article that also has a
PLAIN term, and such an article is found through the PLAIN terms; terms that PMC's case-insensitive search turns into one
of those names or into a name or word seen as noise ("Bert" stays: it is BERT; "hubert", "robert" go); "Gemma"/"Gemini"
glued to lower-case letters (Gemmatimonadetes, Geminiviridae). Per-term body counts: scratch term_counts_body.json.
    python3 gap_body_query.py TERM_COUNTS.json OUT.json
"""
import calendar, json, os, re, sys, time, urllib.parse, urllib.request

sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms                                                     # noqa: E402

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
KEY = os.environ.get("NCBI_API_KEY", "")
WAIT = 0.11 if KEY else 0.35
URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
R1_BODY = ["large language model", "large language models", "Bidirectional encoder representations from transformers",
           "Transformer model", "Transformer models"]
NOEVID = ["BERT", "RoBERTa", "language model", "language models"]
BLOCK = {w.lower() for w in list(focal_terms.GUARDED_NAMES) + list(focal_terms.NEED_EVIDENCE)} | {"grover", "molgen"}
NOISE = {"hubert", "robert", "roberta s", "albert", "herbert", "alberto", "roberto", "norbert", "gilbert", "lambert"}


def esearch(term, a, b):
    d = urllib.parse.urlencode(dict(db="pmc", term=term, retmax=10000, retmode="json", mindate=a, maxdate=b, datetype="pdat",
                                    **({"api_key": KEY} if KEY else {}))).encode()
    for k in range(5):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=d, method="POST"), timeout=180).read())
            time.sleep(WAIT)
            return int(r["esearchresult"]["count"]), r["esearchresult"].get("idlist", [])
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(term[:80])


def ids_for(q):
    out = set()
    for y in range(2015, 2027):
        end = f"{y}/12/31" if y < 2026 else "2026/09/30"
        n, ids = esearch(q, f"{y}/01/01", end)
        if n <= 9999:
            out |= set(ids); continue
        for m in range(1, 13 if y < 2026 else 10):
            last = calendar.monthrange(y, m)[1]
            n2, ids2 = esearch(q, f"{y}/{m:02d}/01", f"{y}/{m:02d}/{last}")
            if n2 <= 9999:
                out |= set(ids2); continue
            for a, b in ((1, 7), (8, 14), (15, 21), (22, last)):
                n3, ids3 = esearch(q, f"{y}/{m:02d}/{a:02d}", f"{y}/{m:02d}/{b:02d}")
                assert n3 <= 9999, (q[:60], y, m, a, n3)
                out |= set(ids3)
    return {f"PMC{i}" for i in out}


def core(translation):
    m = re.match(r'"([^"]+)"\[', translation or "")
    return m.group(1).lower().replace("-", " ").replace("_", " ") if m else ""


if __name__ == "__main__":
    counts = json.load(open(sys.argv[1]))
    r1 = open("/mnt/hum01-rds/Basov/p91688di/query_pmc_entrez.py").read()
    r1_names = re.findall(r'"([^"]+)"\[Title/Abstract\]', r1)
    keep, dropped = {}, {}
    for t in r1_names:
        (dropped if t.lower() in BLOCK else keep)[t] = "R1 name"
    for t in R1_BODY + NOEVID:
        keep[t] = "R1 body phrase / GENERIC without evidence"
    for k, (n, tr) in counts.items():
        s, t = k.split("\t")
        if s not in ("ALIAS", "G4") or t in keep:
            continue
        c = core(tr)
        if c in BLOCK or c in NOISE or re.match(r"(?i)^(gemma|gemini)[a-z]", t):
            dropped[t] = "collides: " + c; continue
        if c in {x.lower() for x in keep}:
            continue                                                     # same search as a kept term
        keep[t] = s
    terms = list(keep)
    print("terms kept", len(terms), "dropped", len(dropped), flush=True)
    res = {"kept": keep, "dropped": dropped, "ids": {}}
    target = set(open(LCS + "target_pmcids.txt").read().split())
    for field in ("Body - All Words", "Title/Abstract"):
        got = set()
        for i in range(0, len(terms), 80):
            q = "(" + " OR ".join(f'"{t}"[{field}]' for t in terms[i:i + 80]) + ') AND "open access"[filter]'
            got |= ids_for(q)
            print(f"{field}: {min(i + 80, len(terms))}/{len(terms)} terms, {len(got):,} articles, {len(got - target):,} not in R1", flush=True)
        res["ids"][field] = sorted(got)
    new = (set(res["ids"]["Body - All Words"]) | set(res["ids"]["Title/Abstract"])) - target
    res["summary"] = dict(body=len(res["ids"]["Body - All Words"]), tiab=len(res["ids"]["Title/Abstract"]), not_in_r1=len(new))
    json.dump(res, open(sys.argv[2], "w"), indent=0)
    print(json.dumps(res["summary"]))
