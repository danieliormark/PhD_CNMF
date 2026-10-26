# RL-200 (owner 2026-10-26, data preparation for the M1 trial): OpenAlex institution records for every institution on
# an authorship of a working-corpus-v2 work (the work chosen as in author_coverage.py --openalex refresh --corpus v2),
# plus every ancestor in their lineage and every parent named in associated_institutions, so that institutions can be
# climbed to their organisation without further requests. Fields: id, display_name, type, country_code, ror, lineage,
# associated_institutions (relationship parent / child / related), display_name_alternatives, display_name_acronyms.
# 50 IDs per request. Key and contact e-mail from the environment only (never written to disk):
#   OPENALEX_API_KEY=... OPENALEX_MAILTO=... python3 institutions_fetch.py
# Cache diagnostics/author_coverage/institutions_openalex.json (keyed by short ID; None for an ID OpenAlex no longer has).
import json, os, time, urllib.parse, urllib.request

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
D, OD = FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
OUT = OD + "institutions_openalex.json"
KEY, MAIL = os.environ["OPENALEX_API_KEY"], os.environ.get("OPENALEX_MAILTO", "")
SELECT = "id,display_name,type,country_code,ror,lineage,associated_institutions,display_name_alternatives,display_name_acronyms"
sid = lambda u: u.rsplit("/", 1)[-1]

wc = [l.strip() for l in open(FP + "corpus_statistics/working_corpus_v2_pmcids.txt") if l.strip()]
H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
V = json.load(open(D + "openalex_verified.json")); V.update(json.load(open(D + "openalex_verified_new.json")))
OW = json.load(open(OD + "openalex_works.json"))


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(sid(w["id"])) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


def call(ids):
    q = {"filter": "openalex_id:" + "|".join(ids), "per-page": 50, "select": SELECT, "api_key": KEY}
    if MAIL:
        q["mailto"] = MAIL
    url = "https://api.openalex.org/institutions?" + urllib.parse.urlencode(q)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)["results"]
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(4 * (attempt + 1))
    raise RuntimeError("request failed")


cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
want = set()
for p in wc:
    for a in (work(p) or {}).get("authorships") or []:
        for i in a.get("institutions") or []:
            if i.get("id"):
                want.add(sid(i["id"]))
print("institutions on authorships:", len(want), flush=True)
rounds = 0
while True:
    todo = sorted(i for i in want if i not in cache)
    if not todo:
        break
    rounds += 1
    print(f"round {rounds}: fetching {len(todo)}", flush=True)
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        got = {sid(r["id"]): r for r in call(chunk)}
        for i in chunk:
            cache[i] = got.get(i)
        if (k // 50) % 20 == 0:
            json.dump(cache, open(OUT, "w"))
    json.dump(cache, open(OUT, "w"))
    for r in [r for r in cache.values() if r]:                          # ancestors and named parents, next round
        want.update(sid(x) for x in r.get("lineage") or [])
        want.update(sid(a["id"]) for a in r.get("associated_institutions") or [] if a.get("relationship") == "parent")
json.dump(cache, open(OUT, "w"))
print("DONE.", sum(1 for v in cache.values() if v), "records,", sum(1 for v in cache.values() if v is None), "gone;", rounds, "rounds")
