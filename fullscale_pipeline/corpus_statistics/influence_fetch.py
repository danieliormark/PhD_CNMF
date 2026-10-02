# Paper metrics + author positions for every matched work, then h-index of every identified
# author. Key from OPENALEX_API_KEY (never written to disk). Resumable caches.
import json, os, time, urllib.parse, urllib.request

D = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/"
KEY = os.environ["OPENALEX_API_KEY"]
V = {k: w for k, (m, w) in json.load(open(D + "openalex_verified.json")).items() if w}
WC, AC = D + "works_metrics.json", D + "author_hindex.json"
works = json.load(open(WC)) if os.path.exists(WC) else {}
auth = json.load(open(AC)) if os.path.exists(AC) else {}


def get(path, params):
    params = dict(params, api_key=KEY)
    url = f"https://api.openalex.org/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)["results"], r.headers.get("x-ratelimit-remaining")
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError("request failed")


short = lambda u: u.rsplit("/", 1)[-1] if u else None
wids = sorted({short(w["id"]) for w in V.values()} - set(works))
for i in range(0, len(wids), 100):
    res, rem = get("works", {"filter": "openalex:" + "|".join(wids[i:i + 100]), "per_page": 200,
                             "select": "id,publication_year,cited_by_count,fwci,citation_normalized_percentile,"
                                       "primary_topic,authorships"})
    for w in res:
        works[short(w["id"])] = dict(
            year=w.get("publication_year"), cited=w.get("cited_by_count"), fwci=w.get("fwci"),
            pct=(w.get("citation_normalized_percentile") or {}).get("value"),
            field=((w.get("primary_topic") or {}).get("field") or {}).get("display_name"),
            authors=[dict(id=short(a["author"].get("id")), pos=a.get("author_position"),
                          raw=a.get("raw_author_name")) for a in w["authorships"]])
    if (i // 100) % 50 == 0:
        json.dump(works, open(WC, "w")); print("works", i + 100, "remaining", rem, flush=True)
    time.sleep(0.05)
json.dump(works, open(WC, "w"))

aids = sorted({a["id"] for w in works.values() for a in w["authors"] if a["id"]} - set(auth))
print("authors to fetch", len(aids), flush=True)
for i in range(0, len(aids), 100):
    res, rem = get("authors", {"filter": "openalex:" + "|".join(aids[i:i + 100]), "per_page": 200,
                               "select": "id,summary_stats,works_count,cited_by_count"})
    for a in res:
        s = a.get("summary_stats") or {}
        auth[short(a["id"])] = dict(h=s.get("h_index"), works=a.get("works_count"), cited=a.get("cited_by_count"))
    if (i // 100) % 200 == 0:
        json.dump(auth, open(AC, "w")); print("authors", i + 100, "remaining", rem, flush=True)
    time.sleep(0.05)
json.dump(auth, open(AC, "w"))
print("DONE works", len(works), "authors", len(auth), "remaining", rem)
