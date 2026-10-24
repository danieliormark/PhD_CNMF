# RL-188 (owner 2026-10-24, M1 trial subset): OpenAlex primary topic (topic -> subfield -> field -> domain, with its
# score) and the topics list of every work in the refreshed works cache (openalex_works.json, current and new
# articles), by work ID, 100 per request. A separate cache, so the works cache keeps the form it was analysed in.
# Key and contact e-mail from the environment only (never written to disk):
#   OPENALEX_API_KEY=... OPENALEX_MAILTO=... python3 openalex_topics_fetch.py
import json, os, time, urllib.parse, urllib.request

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
W = json.load(open(FP + "diagnostics/author_coverage/openalex_works.json"))
OUT = FP + "diagnostics/author_coverage/openalex_topics.json"
KEY, MAIL = os.environ["OPENALEX_API_KEY"], os.environ.get("OPENALEX_MAILTO", "")
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}


def call(params):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(dict(params, api_key=KEY, mailto=MAIL))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(4 * (attempt + 1))
    raise RuntimeError("request failed")


def level(t, k):
    x = (t or {}).get(k) or {}
    return dict(id=x.get("id"), name=x.get("display_name"))


def slim_topic(t):
    return dict(id=t.get("id"), name=t.get("display_name"), score=t.get("score"), subfield=level(t, "subfield"),
                field=level(t, "field"), domain=level(t, "domain"))


ids = sorted({w["id"].rsplit("/", 1)[-1] for store in (W["by_id"], W["by_doi"]) for w in store.values() if w} - set(cache))
print("works:", len(ids), flush=True)
for i in range(0, len(ids), 100):
    batch = ids[i:i + 100]
    res = call({"filter": "openalex_id:" + "|".join(batch), "per_page": 100, "select": "id,primary_topic,topics"})
    got = {w["id"].rsplit("/", 1)[-1]: w for w in res["results"]}
    for b in batch:
        w = got.get(b)
        cache[b] = None if w is None else dict(primary=slim_topic(w["primary_topic"]) if w.get("primary_topic") else None,
                                              topics=[slim_topic(t) for t in w.get("topics") or []])
    if (i // 100) % 20 == 0:
        json.dump(cache, open(OUT, "w")); print(i + len(batch), flush=True)
json.dump(cache, open(OUT, "w"))
print("DONE.", sum(1 for v in cache.values() if v and v["primary"]), "with a primary topic,",
      sum(1 for v in cache.values() if v and not v["primary"]), "without,", sum(1 for v in cache.values() if v is None), "gone")
