"""R1b eligibility (2026-10-19, ticket 107 P1): article types of the retrieval-gap candidates, before any download.
For each candidate PMCID: PMC esummary (PMID, title, journal, dates — the metadata R1's Phase 2 stored) and PubMed
efetch (MEDLINE PublicationType), classified with the same categories as corpus_statistics/article_types_valid.py.
Allowed (owner 2026-10-19): research article, historical article, narrative review, systematic / scoping review /
meta-analysis. Trials, protocols, editorials, letters, datasets, proceedings and others are not eligible. A candidate
with no PMID or no PubMed record is kept as "type unknown" and decided after download (PMC header), not dropped here.
Resumable caches in LCS/. Writes LCS/gap_eligibility_<tag>.tsv and LCS/gap_pmcids_<tag>_eligible.txt.
    python3 gap_eligibility.py CANDIDATES.txt TAG          (NCBI_API_KEY from the environment)
"""
import json, os, re, sys, time, urllib.parse, urllib.request

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
KEY = os.environ.get("NCBI_API_KEY", "")
WAIT = 0.11 if KEY else 0.35
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ALLOWED = {"Research article (Journal Article tag, no more specific type)", "Historical article", "Review (narrative / other)",
           "Systematic review / meta-analysis / scoping review"}


def post(endpoint, params):
    params = dict(params, **({"api_key": KEY} if KEY else {}))
    for attempt in range(6):
        try:
            req = urllib.request.Request(EU + endpoint, data=urllib.parse.urlencode(params).encode(), method="POST")
            out = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace")
            time.sleep(WAIT)
            return out
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(3 * (attempt + 1))
    raise RuntimeError(endpoint)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


if __name__ == "__main__":
    cand_p, tag = sys.argv[1], sys.argv[2]
    cand = [l.strip() for l in open(cand_p) if l.strip()]
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "..", "corpus_statistics", "article_types_valid.py")).read()
    ns = {}; exec(src[src.index("ADMIN_TAGS ="):src.index("rows = []")], ns)
    SUM_P, PT_P = LCS + "gap_esummary_cache.json", LCS + "gap_pubtypes_cache.json"
    S, PT = load(SUM_P), load(PT_P)
    todo = [p for p in cand if p not in S]
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        r = json.loads(post("esummary.fcgi", dict(db="pmc", id=",".join(c[3:] for c in chunk), retmode="json")))["result"]
        for c in chunk:
            d = r.get(c[3:]) or {}
            ids = {a.get("idtype"): a.get("value") for a in d.get("articleids", [])}
            S[c] = dict(pmid=ids.get("pmid") or None, doi=ids.get("doi"), title=d.get("title"), source=d.get("source"),
                        fulljournalname=d.get("fulljournalname"), pubdate=d.get("pubdate"), epubdate=d.get("epubdate"))
        if (i // 200) % 20 == 0:
            json.dump(S, open(SUM_P, "w")); print(f"esummary {i + len(chunk)}/{len(todo)}", flush=True)
    json.dump(S, open(SUM_P, "w"))
    PT_RE = re.compile(r'<PublicationType UI="(D\d+)">([^<]*)</PublicationType>')
    PMID_RE = re.compile(r"<PMID[^>]*>(\d+)</PMID>")
    ART_RE = re.compile(r"<PubmedArticle>.*?</PubmedArticle>", re.S)
    pm_todo = sorted({S[c]["pmid"] for c in cand if S[c].get("pmid")} - set(PT))
    for i in range(0, len(pm_todo), 200):
        chunk = pm_todo[i:i + 200]
        xml = post("efetch.fcgi", dict(db="pubmed", id=",".join(chunk), rettype="xml", retmode="xml"))
        found = set()
        for a in ART_RE.findall(xml):
            m = PMID_RE.search(a)
            if m:
                PT[m.group(1)] = [{"ui": u, "name": n} for u, n in PT_RE.findall(a)]; found.add(m.group(1))
        for p in chunk:
            PT.setdefault(p, None)
        if (i // 200) % 20 == 0:
            json.dump(PT, open(PT_P, "w")); print(f"efetch {i + len(chunk)}/{len(pm_todo)}", flush=True)
    json.dump(PT, open(PT_P, "w"))
    rows, eligible = [], []
    for c in cand:
        pm = S[c].get("pmid")
        if not pm:
            cat = "type unknown (no PMID)"
        elif PT.get(pm) is None:
            cat = "type unknown (no PubMed record)"
        else:
            names = [t["name"] for t in PT[pm]]
            cat = ns["classify"](names) if names else "type unknown (no PublicationType)"
        ok = cat in ALLOWED or cat.startswith("type unknown")
        rows.append((c, pm or "", cat, "eligible" if ok else "not eligible"))
        if ok:
            eligible.append(c)
    import collections
    with open(LCS + f"gap_eligibility_{tag}.tsv", "w") as f:
        f.write("pmcid\tpmid\tprimary_type\teligibility\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    open(LCS + f"gap_pmcids_{tag}_eligible.txt", "w").write("\n".join(eligible) + "\n")
    print(json.dumps(collections.Counter(r[2] for r in rows).most_common(), indent=0))
    print("eligible", len(eligible), "of", len(cand))
