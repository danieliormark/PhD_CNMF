# Ticket 107 P1.B (owner, 2026-10-19): institutions of the trial slices (2020, 2021) mapped to the top of their
# OpenAlex lineage (owner: use the higher-level entity, for a denser institutional network). Fetches
# id, display_name, type, country, ror, lineage and associated_institutions for each institution seen on a trial
# byline (OpenAlex /institutions, 50 IDs per request, polite pool via mailto; cached). Reports how many
# institutions merge, the "related" (not parent-child) pairs left separate, and same-name candidates.
#   OPENALEX_MAILTO=... python3 trial_institution_lineage.py
import collections, json, os, time, urllib.parse, urllib.request

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OD = FP + "diagnostics/author_coverage/"
CACHE = OD + "trial_institutions_openalex.json"
OUT = OD + "trial_institution_lineage.json"
P1 = json.load(open(OD + "trial_p1_checks.json"))
TRIAL = {y: set(v) for y, v in P1["trial_pmcids"].items()}
TRIAL["2021"] |= {"PMC7815194", "PMC8051288", "PMC8523728", "PMC8666839", "PMC8752034"}   # E: no-PMID articles, owner-approved
D = FP + "diagnostics/openalex_authors/"
H, V = json.load(open(D + "headers.json")), json.load(open(D + "openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    bd = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return bd if bd and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


# institution occurrences per slice: (article, institution) from every OpenAlex authorship of trial articles
occ = {y: collections.Counter() for y in TRIAL}
for y, arts in TRIAL.items():
    for p in arts:
        w = work(p)
        for a in (w or {}).get("authorships") or []:
            for i in a.get("institutions") or []:
                if i.get("id"):
                    occ[y][i["id"].rsplit("/", 1)[-1]] += 1
ids = sorted(set().union(*(set(c) for c in occ.values())))

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [i for i in ids if i not in cache]
mailto = os.environ.get("OPENALEX_MAILTO", "")
for k in range(0, len(todo), 50):
    chunk = todo[k:k + 50]
    q = {"filter": "openalex_id:" + "|".join(chunk), "per-page": 50,
         "select": "id,display_name,type,country_code,ror,lineage,associated_institutions"}
    if mailto:
        q["mailto"] = mailto
    url = "https://api.openalex.org/institutions?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            res = json.load(urllib.request.urlopen(url, timeout=60))["results"]
            break
        except Exception as e:
            time.sleep(2 ** attempt); res = None
    for r in res or []:
        cache[r["id"].rsplit("/", 1)[-1]] = r
    json.dump(cache, open(CACHE, "w"))
    time.sleep(0.2)
missing = [i for i in ids if i not in cache]

sid = lambda u: u.rsplit("/", 1)[-1]
top = {}
for i in ids:
    lin = [sid(x) for x in (cache.get(i) or {}).get("lineage") or [i]]
    top[i] = lin[-1] if lin else i     # OpenAlex lists lineage from the institution itself up to its root
# names of top-level entities (fetch any not yet cached)
extra = sorted({t for t in top.values() if t not in cache})
for k in range(0, len(extra), 50):
    chunk = extra[k:k + 50]
    q = {"filter": "openalex_id:" + "|".join(chunk), "per-page": 50,
         "select": "id,display_name,type,country_code,ror,lineage,associated_institutions"}
    if mailto:
        q["mailto"] = mailto
    for r in json.load(urllib.request.urlopen("https://api.openalex.org/institutions?" + urllib.parse.urlencode(q), timeout=60))["results"]:
        cache[sid(r["id"])] = r
json.dump(cache, open(CACHE, "w"))
name = lambda i: (cache.get(i) or {}).get("display_name", i)

print(f"Institutions on trial bylines: {len(ids)}; fetched {len(ids) - len(missing)}; missing {len(missing)}")
for y in sorted(TRIAL):
    a = set(occ[y]); t = {top[i] for i in a}
    print(f"  {y}: {len(a)} institutions -> {len(t)} top-level entities ({len(a) - len(t)} merged)")
merged = collections.defaultdict(list)
for i in ids:
    if top[i] != i:
        merged[top[i]].append(i)
print(f"\nInstitutions folded into a parent: {sum(len(v) for v in merged.values())}, into {len(merged)} parents. Examples:")
for t, kids in sorted(merged.items(), key=lambda kv: -len(kv[1]))[:12]:
    print(f"  {name(t)}  <-  " + "; ".join(name(k) for k in kids[:4]) + (" ..." if len(kids) > 4 else ""))
types = collections.Counter((cache.get(top[i]) or {}).get("type") for i in ids)
print("\nTop-level entity types:", dict(types))
# "related" associations between two top-level entities both present (left separate by the lineage rule)
tops = set(top.values())
rel = set()
for t in tops:
    for a in (cache.get(t) or {}).get("associated_institutions") or []:
        o = sid(a["id"])
        if o in tops and a.get("relationship") == "related":
            rel.add(tuple(sorted((t, o))))
print(f"\n'Related' (not parent-child) pairs among top-level entities, left separate: {len(rel)}. Examples:")
for x, z in sorted(rel)[:10]:
    print(f"  {name(x)}  ~  {name(z)}")
norm = collections.defaultdict(set)
for t in tops:
    norm[name(t).lower().replace("the ", "").strip()].add(t)
same = {n: s for n, s in norm.items() if len(s) > 1}
print(f"\nTop-level entities sharing a display name: {len(same)} {list(same.items())[:5]}")
json.dump(dict(top=top, merged={k: v for k, v in merged.items()}, related=sorted(rel), same_name={k: sorted(v) for k, v in same.items()},
               missing=missing, per_slice={y: dict(institutions=len(set(occ[y])), top=len({top[i] for i in occ[y]})) for y in TRIAL}),
          open(OUT, "w"), indent=1)
