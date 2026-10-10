# Fresh OpenAlex work records for the working corpus (working_corpus_pmcids.txt), full authorships
# (author ID, ORCID, raw name, institutions, raw affiliation strings) and locations. Same version rule as
# verify.py: per article, the record whose DOI is the article's own PMC-header DOI; the work chosen there
# (openalex_verified.json) is refreshed by its ID, the DOI is looked up again where verify.py found no
# record or another publication, and works with 100 authorships (the list endpoint's cut) are re-read
# one by one. Key and contact e-mail come from the environment only (never written to disk):
#   OPENALEX_API_KEY=... OPENALEX_MAILTO=... python3 openalex_works_fetch.py
import json, os, time, urllib.parse, urllib.request

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
D = FP + "diagnostics/openalex_authors/"
OUT = FP + "diagnostics/author_coverage/openalex_works.json"
KEY, MAIL = os.environ["OPENALEX_API_KEY"], os.environ.get("OPENALEX_MAILTO", "")
H = json.load(open(D + "headers.json"))
V = json.load(open(D + "openalex_verified.json"))
wc = [l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()]
SELECT = "id,doi,ids,title,publication_year,type,primary_location,locations,authorships"
norm_doi = lambda d: (d or "").lower().replace("https://doi.org/", "").strip()


def call(path, params):
    params = dict(params, api_key=KEY, mailto=MAIL)
    url = "https://api.openalex.org/" + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print("retry", attempt, "HTTP", e.code, flush=True)
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True)
        time.sleep(4 * (attempt + 1))
    raise RuntimeError("request failed: " + path)


def slim_src(loc):
    s = (loc or {}).get("source") or {}
    return dict(id=s.get("id"), name=s.get("display_name"), type=s.get("type"), issn_l=s.get("issn_l"),
                version=(loc or {}).get("version"), is_oa=(loc or {}).get("is_oa"))


def slim(w):
    return dict(id=w["id"], doi=norm_doi(w.get("doi")), pmid=((w.get("ids") or {}).get("pmid") or "").rsplit("/", 1)[-1],
                title=w.get("title"), year=w.get("publication_year"), type=w.get("type"),
                primary_location=slim_src(w.get("primary_location")),
                locations=[slim_src(l) for l in w.get("locations") or []],
                authorships=[dict(pos=a.get("author_position"), id=(a.get("author") or {}).get("id"),
                                  orcid=(a.get("author") or {}).get("orcid"), name=(a.get("author") or {}).get("display_name"),
                                  raw=a.get("raw_author_name"), corresponding=a.get("is_corresponding"),
                                  institutions=[dict(id=i.get("id"), name=i.get("display_name"), ror=i.get("ror"),
                                                     country=i.get("country_code"), type=i.get("type"))
                                                for i in a.get("institutions") or []],
                                  raw_affiliations=a.get("raw_affiliation_strings") or [])
                             for a in w.get("authorships") or []])


cache = json.load(open(OUT)) if os.path.exists(OUT) else {"by_id": {}, "by_doi": {}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 1. refresh the works verify.py chose
ids = sorted({V[p][1]["id"].rsplit("/", 1)[-1] for p in wc if V.get(p) and V[p][1]} - set(cache["by_id"]))
print("works to refresh:", len(ids), flush=True)
for i in range(0, len(ids), 100):
    batch = ids[i:i + 100]
    res = call("works", {"filter": "openalex_id:" + "|".join(batch), "per_page": 100, "select": SELECT})
    got = {w["id"].rsplit("/", 1)[-1]: slim(w) for w in res["results"]}
    for b in batch:
        cache["by_id"][b] = got.get(b)          # None: merged away or deleted since 2026-10-02
    if (i // 100) % 20 == 0:
        json.dump(cache, open(OUT, "w")); print(i + len(batch), flush=True)

# 2. DOI lookups: no record before, another publication before, or a refreshed work that vanished
need = set()
for p in wc:
    m, w = V.get(p) or ["unmatched", None]
    hd = norm_doi((H.get(p) or {}).get("doi"))
    gone = w and cache["by_id"].get(w["id"].rsplit("/", 1)[-1]) is None
    if hd and (not w or m == "pmid_doi_mismatch_no_doi_hit" or gone):
        need.add(hd)
need = sorted(need - set(cache["by_doi"]))
print("DOI lookups:", len(need), flush=True)
for i in range(0, len(need), 50):
    batch = need[i:i + 50]
    res = call("works", {"filter": "doi:" + "|".join(batch), "per_page": 100, "select": SELECT})
    got = {}
    for w in res["results"]:
        got.setdefault(norm_doi(w.get("doi")), slim(w))
    for d in batch:
        cache["by_doi"][d] = got.get(d)

# 3. re-read works whose list-endpoint authorships stop at 100
full = [k for k, w in cache["by_id"].items() if w and len(w["authorships"]) == 100 and not w.get("full")]
full += [d for d, w in cache["by_doi"].items() if w and len(w["authorships"]) == 100 and not w.get("full")]
print("works with 100 authorships, re-read singly:", len(full), flush=True)
for k in full:
    store = cache["by_id"] if k in cache["by_id"] else cache["by_doi"]
    wid = store[k]["id"].rsplit("/", 1)[-1]
    w = call("works/" + wid, {"select": SELECT})
    if w:
        s = slim(w); s["full"] = True; store[k] = s

cache["fetched"] = time.strftime("%Y-%m-%d %H:%M")
json.dump(cache, open(OUT, "w"))
print("DONE. works:", sum(1 for w in cache["by_id"].values() if w), "refreshed,",
      sum(1 for w in cache["by_id"].values() if w is None), "gone;",
      sum(1 for w in cache["by_doi"].values() if w), "of", len(cache["by_doi"]), "DOI lookups found")
