# Group 4 (owner 2026-10-19): unlisted model-name forms, counted only in articles of the allowed types
# (research article, historical article, narrative review, systematic / scoping review / meta-analysis).
import collections, json, re, sys
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics")
R = json.load(open(sys.argv[1]))
D = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/"
HDR = json.load(open(D + "headers.json")); PT = json.load(open(D + "pubmed_pubtypes.json"))
src = open("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/article_types_valid.py").read()
ns = {}; exec(src[src.index("ADMIN_TAGS ="):src.index("rows = []")], ns)
ALLOWED = {"Research article (Journal Article tag, no more specific type)", "Historical article", "Review (narrative / other)",
           "Systematic review / meta-analysis / scoping review"}
def typ(p):
    pm = (HDR.get(p) or {}).get("pmid")
    if not pm or str(pm) not in PT: return "unknown (not in the cache)"
    names = [t["name"] for t in PT[str(pm)]]
    return ns["classify"](names) if names else "unclassified"
FAM = r"(?:BERT|RoBERTa|GPT|LLaMA|Llama|LLAMA|PaLM|Gemini|Mistral|Gemma|Bard)"
LOWFAM = r"(?:bert|roberta|gpt|llama|gemini|mistral|gemma|palm|bard)"
H = r"[-‐‑]"
RULES = [
    ("A1 hyphen prefix + listed family name (CT-BERT, Med-PaLM)", re.compile(rf"^[A-Za-z0-9.+]+{H}{FAM}(?:{H}.*|\d.*)?$")),
    ("A2 version glued to a listed name (ChatGPT4, Gemini2.5)", re.compile(rf"^(?:ChatGPT|Chat-GPT|GPT|Gemini|PaLM|Gemma|Mistral|LLaMA|Llama|Claude|Grok)\d[0-9.]*[a-zA-Z]{{0,6}}$")),
    ("A3 plural of a listed name (GPTs, BERTs)", re.compile(rf"^(?:{FAM}|ChatGPT|LLM)s$")),
    ("A4 listed name in another case (bert, Bert, LLAMA)", re.compile(rf"(?i)^{LOWFAM}$|^roberta$")),
    ("A5 checkpoint name: listed name + suffix (bert-base-uncased)", re.compile(rf"(?i)^(?:{LOWFAM}|roberta|chatgpt){H}[A-Za-z0-9.]+(?:{H}[A-Za-z0-9.]+)*$")),
    ("A6 hyphenated 'language-model'", re.compile(r"(?i)^(?:.*[-‐‑_])?language[-‐‑_]model(?:s|ing|ling|led)?(?:[-‐‑_].*)?$")),
    ("B glued derivative (DistilBERT, scGPT, MedGemma)", re.compile(rf"(?:[A-Za-z0-9]{FAM}|{FAM}[a-z])")),
]


def cls(tok):
    for name, rx in RULES:
        if rx.search(tok):
            return name
    return "C other (mostly noise: Hilbert, puberty, palmitate)"



T = collections.Counter(); forms = collections.defaultdict(lambda: [0, set(), set()])   # occ, articles, new-candidate articles
for a, r in R.items():
    t = typ(a); T[t] += 1
    if t not in ALLOWED and r["status"] == "ok": continue
    for o in r["occ"]:
        c = cls(o["tok"])
        if not (c.startswith("B") or c.startswith("C")): continue
        f = forms[o["tok"]]; f[0] += 1; f[1].add(a)
        if r["status"] != "ok": f[2].add(a)
print("articles with candidates, by type:", T.most_common())
rows = sorted(((len(v[1]), v[0], len(v[2]), k) for k, v in forms.items()), reverse=True)
print("forms", len(rows), "occurrences", sum(r[1] for r in rows))
cum = 0; tot = sum(r[1] for r in rows)
for i, (na, n, nn, k) in enumerate(rows):
    cum += n
    if i < 400: print(f"{k}\t{na}\t{n}\t{nn}")
    if i in (99, 199, 399, 799): print(f"--- top {i+1} forms cover {100*cum/tot:.1f}% of occurrences")
json.dump([dict(form=k, articles=na, occ=n, new_articles=nn) for na, n, nn, k in rows], open("g4_forms.json", "w"))
