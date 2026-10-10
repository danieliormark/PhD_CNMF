# MEDLINE PublicationTypeList for every G2-parsed article's PMID (NLM-assigned article-type
# metadata, independent of the publisher's own self-declared JATS subj-group text that the
# "Subjects:" field in headers.json comes from -- see CORPUS_STATISTICS.md §2 for why that text
# field is not reliable for the review/research split). Resumable cache, batched efetch POSTs.
import json, os, time, urllib.request, urllib.parse, re

D = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/"
KEY = os.environ.get("NCBI_API_KEY")
H = json.load(open(D + "headers.json"))
pmid_to_pmcids = {}
for pmcid, v in H.items():
    if v and v.get("pmid"):
        pmid_to_pmcids.setdefault(v["pmid"], []).append(pmcid)
pmids = sorted(pmid_to_pmcids)
print("distinct PMIDs to fetch:", len(pmids))

OUT = D + "pubmed_pubtypes.json"
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
todo = [p for p in pmids if p not in cache]
print("already cached:", len(cache), "remaining:", len(todo))

PT_RE = re.compile(r'<PublicationType UI="(D\d+)">([^<]*)</PublicationType>')
PMID_RE = re.compile(r"<PMID[^>]*>(\d+)</PMID>")
ART_RE = re.compile(r"<PubmedArticle>.*?</PubmedArticle>", re.S)

BATCH = 200
for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    params = {"db": "pubmed", "id": ",".join(chunk), "rettype": "xml", "retmode": "xml"}
    if KEY:
        params["api_key"] = KEY
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                xml = r.read().decode("utf-8", "replace")
            break
        except Exception as e:
            print("retry", attempt, type(e).__name__, e, flush=True)
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"batch at {i} failed after retries")

    found = set()
    for art_xml in ART_RE.findall(xml):
        m = PMID_RE.search(art_xml)
        if not m:
            continue
        pmid = m.group(1)
        pts = PT_RE.findall(art_xml)
        cache[pmid] = [{"ui": ui, "name": name} for ui, name in pts]
        found.add(pmid)
    for p in chunk:
        if p not in found:
            cache[p] = None  # not found / withdrawn / PMID mismatch

    if (i // BATCH) % 10 == 0:
        json.dump(cache, open(OUT, "w"))
        print(f"{i + len(chunk)}/{len(todo)} fetched, {len(found)} found this batch", flush=True)
    time.sleep(0.12 if KEY else 0.35)

json.dump(cache, open(OUT, "w"))
ok = [p for p in pmids if cache.get(p)]
none_pt = [p for p in pmids if p in cache and cache[p] == []]
missing = [p for p in pmids if cache.get(p) is None]
print(f"DONE. {len(pmids)} PMIDs: {len(ok)} with >=1 PublicationType, {len(none_pt)} found but empty PT list, "
      f"{len(missing)} not found/withdrawn")
