# RL-193 (owner 2026-10-26, M1 trial subset by discipline): NLM Catalog subject data for every journal of the working
# corpus v2 (NLM journal ID from PubMed, pubmed_authors.json), as an independent check on OpenAlex's topic fields.
# Per journal: Broad Subject Terms (esummary "broadheading", NLM's ~120 journal-level subject categories), the
# catalog's MeSH subject headings (efetch MeshHeadingList, with the major-topic flag), title, current indexing status.
# NLM IDs ending in "R", starting with "0" or in the 16-digit form (9918...6676) are not catalog UIDs and are resolved by esearch [nlmid] first. Read-only Entrez; optional
# NCBI_API_KEY from the environment. Resumable cache diagnostics/author_coverage/nlm_journals.json (keyed by NLM ID).
import json, os, re, time, urllib.parse, urllib.request

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OUT = FP + "diagnostics/author_coverage/nlm_journals.json"
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
KEY = os.environ.get("NCBI_API_KEY")
PM = json.load(open(FP + "diagnostics/author_coverage/pubmed_authors.json"))
wanted = sorted({v["journal"]["nlm_id"] for v in PM.values() if v and v.get("journal") and v["journal"].get("nlm_id")})
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}


def call(fn, params):
    params = dict(params, **({"api_key": KEY} if KEY else {}))
    for attempt in range(6):
        try:
            req = urllib.request.Request(EU + fn, data=urllib.parse.urlencode(params).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                out = r.read().decode("utf-8", "replace")
            time.sleep(0.12 if KEY else 0.4)
            return out
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(4 * (attempt + 1))
    raise RuntimeError("request failed")


todo = [n for n in wanted if n not in cache]
print("journals:", len(wanted), "to fetch:", len(todo), flush=True)
direct = lambda n: n.isdigit() and not n.startswith("0") and len(n) <= 9
uid = {n: n for n in todo if direct(n)}
for n in todo:
    if not direct(n):
        ids = re.findall(r"<Id>(\d+)</Id>", call("esearch.fcgi", {"db": "nlmcatalog", "term": f"{n}[nlmid]"}))
        uid[n] = ids[0] if len(ids) == 1 else None
        if len(ids) != 1:
            cache[n] = dict(status=f"esearch returned {len(ids)} records")
by_uid = {u: n for n, u in uid.items() if u}
MESH = re.compile(r'<DescriptorName MajorTopicYN="([YN])">([^<]*)</DescriptorName>')
for i in range(0, len(by_uid), 200):
    us = list(by_uid)[i:i + 200]
    summ = json.loads(call("esummary.fcgi", {"db": "nlmcatalog", "id": ",".join(us), "retmode": "json"}))["result"]
    xml = call("efetch.fcgi", {"db": "nlmcatalog", "id": ",".join(us), "retmode": "xml"})
    mesh = {}
    for rec in re.findall(r"<NLMCatalogRecord[ >].*?</NLMCatalogRecord>", xml, re.S):
        m = re.search(r"<NlmUniqueID>([^<]+)</NlmUniqueID>", rec)
        if m:
            mesh[m.group(1)] = [dict(term=t, major=f == "Y") for f, t in MESH.findall(rec)]
    for u in us:
        n, s = by_uid[u], summ.get(u) or {}
        if not s or s.get("error"):
            cache[n] = dict(status="no summary"); continue
        if s.get("nlmuniqueid") != n:
            cache[n] = dict(status=f"summary for another ID {s.get('nlmuniqueid')}"); continue
        cache[n] = dict(status="ok", uid=u, title=(s.get("titlemainlist") or [{}])[0].get("title"),
                        ta=s.get("medlineta"), indexing=s.get("currentindexingstatus"),
                        broad=s.get("broadheading") or [], mesh=mesh.get(n, []))
    json.dump(cache, open(OUT, "w")); print(i + len(us), flush=True)
json.dump(cache, open(OUT, "w"))
ok = [v for v in cache.values() if v.get("status") == "ok"]
print("DONE.", len(ok), "ok,", sum(1 for v in ok if v["broad"]), "with a Broad Subject Term,",
      sum(1 for v in ok if v["mesh"]), "with MeSH;", len(cache) - len(ok), "not resolved")
