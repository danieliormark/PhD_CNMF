# PubMed (MEDLINE citation XML, NCBI Entrez efetch) author list, affiliations, journal and dates
# for every working-corpus article with a PMID (working_corpus_pmcids.txt; CORPUS_STATISTICS.md §8-§9).
# The byline backbone for the author coverage analysis: name, order, ORCID where the publisher
# supplied it, per-author affiliation text, collective (group) names. Resumable cache, batched POSTs.
#   NCBI_API_KEY=... python3 pubmed_authors_fetch.py
import json, os, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
H = json.load(open(FP + "diagnostics/openalex_authors/headers.json"))
OUTDIR = FP + "diagnostics/author_coverage/"
OUT = OUTDIR + "pubmed_authors.json"
KEY = os.environ.get("NCBI_API_KEY")

wc = [l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()]
pmids = sorted({H[p]["pmid"] for p in wc if (H.get(p) or {}).get("pmid")})
os.makedirs(OUTDIR, exist_ok=True)
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
todo = [p for p in pmids if p not in cache]
print("PMIDs:", len(pmids), "cached:", len(cache), "to fetch:", len(todo), flush=True)


def txt(e, path):
    x = e.find(path)
    return "".join(x.itertext()).strip() if x is not None else None


def parse(art):
    mc = art.find("MedlineCitation")
    a = mc.find("Article")
    j = a.find("Journal")
    al = a.find("AuthorList")
    authors = []
    for au in (al.findall("Author") if al is not None else []):
        authors.append(dict(
            valid=au.get("ValidYN", "Y"),
            last=txt(au, "LastName"), fore=txt(au, "ForeName"), initials=txt(au, "Initials"),
            collective=txt(au, "CollectiveName"),
            ids=[(i.get("Source"), (i.text or "").strip()) for i in au.findall("Identifier")],
            affs=[dict(text=txt(ai, "Affiliation"),
                       ids=[(i.get("Source"), (i.text or "").strip()) for i in ai.findall("Identifier")])
                  for ai in au.findall("AffiliationInfo")]))
    issns = [(i.get("IssnType"), (i.text or "").strip()) for i in j.findall("ISSN")]
    mji = mc.find("MedlineJournalInfo")
    ad = a.find("ArticleDate")
    return dict(
        complete=al.get("CompleteYN", "Y") if al is not None else None,
        authors=authors,
        journal=dict(title=txt(j, "Title"), iso=txt(j, "ISOAbbreviation"), issn=issns,
                     nlm_id=txt(mji, "NlmUniqueID") if mji is not None else None,
                     ta=txt(mji, "MedlineTA") if mji is not None else None),
        pub_year=txt(j, "JournalIssue/PubDate/Year") or (txt(j, "JournalIssue/PubDate/MedlineDate") or "")[:4] or None,
        epub_year=txt(ad, "Year") if ad is not None else None,
    )


BATCH = 200
for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}
    if KEY:
        params["api_key"] = KEY
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                                         data=urllib.parse.urlencode(params).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                root = ET.fromstring(r.read())
            break
        except Exception as e:
            print("retry", attempt, type(e).__name__, e, flush=True)
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"batch at {i} failed after retries")
    found = set()
    for art in root.findall("PubmedArticle"):
        pmid = art.find("MedlineCitation/PMID").text.strip()
        if pmid in chunk:
            cache[pmid] = parse(art)
            found.add(pmid)
    for p in chunk:
        if p not in found:
            cache[p] = None
    if (i // BATCH) % 10 == 0:
        json.dump(cache, open(OUT, "w"))
        print(f"{i + len(chunk)}/{len(todo)}", flush=True)
    time.sleep(0.12 if KEY else 0.35)

json.dump(cache, open(OUT, "w"))
print("DONE.", sum(1 for p in pmids if cache.get(p)), "found,", sum(1 for p in pmids if cache.get(p) is None), "not found")
