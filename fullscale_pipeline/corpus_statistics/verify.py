# Verified OpenAlex match: every work returned per PMID is kept, then checked against the
# DOI in the article's own PMC header; unmatched articles retried by DOI. Key read from the
# OPENALEX_API_KEY environment variable (never written to disk).
import json, os, time, urllib.parse, urllib.request, collections

D = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/"
H = json.load(open(D + "headers.json"))
KEY = os.environ["OPENALEX_API_KEY"]
RAW = D + "openalex_raw.json"
raw = json.load(open(RAW)) if os.path.exists(RAW) else {"by_pmid": {}, "by_doi": {}}
norm_doi = lambda d: (d or "").lower().replace("https://doi.org/", "").strip()


def get(filt):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": filt, "per_page": 200, "select": "id,ids,doi,title,authorships", "api_key": KEY})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["results"], r.headers.get("x-ratelimit-remaining")
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError("request failed")


def slim(w):
    return dict(id=w["id"], doi=norm_doi(w.get("doi")), pmid=(w.get("ids", {}).get("pmid") or "").rsplit("/", 1)[-1],
                title=w.get("title"), authors=[dict(id=a["author"].get("id"), orcid=a["author"].get("orcid"),
                                                    raw=a.get("raw_author_name")) for a in w["authorships"]])


pmids = sorted({v["pmid"] for v in H.values() if v and v["pmid"]} - set(raw["by_pmid"]))
for i in range(0, len(pmids), 100):
    batch = pmids[i:i + 100]
    res, rem = get("ids.pmid:" + "|".join(batch))
    got = collections.defaultdict(list)
    for w in res:
        s = slim(w); got[s["pmid"]].append(s)
    for p in batch:
        raw["by_pmid"][p] = got.get(p, [])
    if (i // 100) % 50 == 0:
        json.dump(raw, open(RAW, "w")); print(i + len(batch), "pmids; remaining", rem, flush=True)
    time.sleep(0.1)

# Choose per article: a PMID hit whose DOI agrees with the header DOI; else DOI lookup.
need_doi = []
choice = {}
for pmcid, h in H.items():
    hd = norm_doi(h["doi"]) if h else ""
    cands = raw["by_pmid"].get(h["pmid"], []) if h and h["pmid"] else []
    agree = [c for c in cands if hd and c["doi"] == hd]
    if agree:
        choice[pmcid] = ("pmid+doi", agree[0])
    elif cands and not hd:
        choice[pmcid] = ("pmid_only_no_header_doi", cands[0])
    elif hd:
        need_doi.append((pmcid, hd, cands))
    else:
        choice[pmcid] = ("unmatched", None)
dois = sorted({d for _, d, _ in need_doi} - set(raw["by_doi"]))
for i in range(0, len(dois), 50):
    batch = dois[i:i + 50]
    res, rem = get("doi:" + "|".join(batch))
    got = collections.defaultdict(list)
    for w in res:
        s = slim(w); got[s["doi"]].append(s)
    for d in batch:
        raw["by_doi"][d] = got.get(d, [])
json.dump(raw, open(RAW, "w"))
for pmcid, hd, cands in need_doi:
    via_doi = raw["by_doi"].get(hd, [])
    if via_doi:
        choice[pmcid] = ("doi" if not cands else "doi_differs_from_pmid_hit", via_doi[0])
    elif cands:
        choice[pmcid] = ("pmid_doi_mismatch_no_doi_hit", cands[0])
    else:
        choice[pmcid] = ("unmatched", None)

json.dump({k: (m, w) for k, (m, w) in choice.items()}, open(D + "openalex_verified.json", "w"))
multi = sum(1 for v in raw["by_pmid"].values() if len(v) > 1)
print("PMIDs returning >1 OpenAlex work:", multi, "of", len(raw["by_pmid"]))
print("match types:", dict(collections.Counter(m for m, _ in choice.values())))
print("remaining credits", rem)
