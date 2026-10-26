# RL-200 (owner 2026-10-26): can bylines without an OpenAlex institution ID be given an institution from their affiliation
# text? Read-only test on the Psychology trial list; nothing is applied. Two methods, each on every distinct affiliation
# string (PubMed's; OpenAlex raw affiliation if PubMed has none) of a byline whose aligned OpenAlex authorship has no
# institution ID:
#   (1) ROR affiliation matching (api.ror.org/v2/organizations?affiliation=..., only the match ROR marks "chosen"), the
#       ROR ID mapped to an OpenAlex institution (cache ror field, else OpenAlex filter ror:); cache
#       diagnostics/author_coverage/ror_affiliation_cache.json (network only for strings not cached);
#   (2) in-corpus name match: an institution of the corpus cache (institutions_openalex.json) whose display name or listed
#       alternative name (2+ words, not only generic words) occurs in the string; longest name wins; unique only; country
#       must agree when the string names one of a short list of countries.
# Result is printed; precision was judged by reading every match (RUN_LOG RL-200).
#   OPENALEX_API_KEY=... python3 affiliation_text_match_test.py
import collections, json, os, re, time, unicodedata, urllib.parse, urllib.request

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
D, OD = FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
sid = lambda u: u.rsplit("/", 1)[-1]
psy = {l.strip() for l in open(FP + "corpus_statistics/psychology_trial_pmcids.txt") if l.strip()}
B = [b for b in map(json.loads, open(OD + "bylines_v2_refresh.jsonl")) if b["pmcid"] in psy]
H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
V = json.load(open(D + "openalex_verified.json")); V.update(json.load(open(D + "openalex_verified_new.json")))
OW, PM = json.load(open(OD + "openalex_works.json")), json.load(open(OD + "pubmed_authors.json"))
C = json.load(open(OD + "institutions_openalex.json"))


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(sid(w["id"])) if w else None
    bd = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return bd if bd and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


texts = collections.Counter()
for b in B:
    au = (work(b["pmcid"]) or {}).get("authorships") or []
    a = au[b["oa_idx"]] if b["aligned"] and b["oa_idx"] is not None and b["oa_idx"] < len(au) else None
    if [i for i in (a or {}).get("institutions") or [] if i.get("id")]:
        continue
    pa = (PM.get((H.get(b["pmcid"]) or {}).get("pmid") or "") or {}).get("authors") or []
    pmb = pa[b["pos"]] if b["pos"] < len(pa) else {}
    for t in [x["text"] for x in pmb.get("affs") or []] or ((a or {}).get("raw_affiliations") or []):
        texts[t] += 1
# (1) ROR
cp = OD + "ror_affiliation_cache.json"; R = json.load(open(cp)) if os.path.exists(cp) else {}
for t in texts:
    if t in R:
        continue
    r = {"items": []}
    for k in range(5):
        try:
            r = json.load(urllib.request.urlopen("https://api.ror.org/v2/organizations?affiliation=" + urllib.parse.quote(t[:500]), timeout=60)); break
        except Exception:
            time.sleep(3 * (k + 1))
    ch = [x for x in r.get("items", []) if x.get("chosen")]
    R[t] = dict(ror=ch[0]["organization"]["id"] if ch else None, score=ch[0]["score"] if ch else None,
                name=([n["value"] for n in ch[0]["organization"]["names"] if "ror_display" in n.get("types", [])] or [""])[0] if ch else None)
    time.sleep(0.15)
json.dump(R, open(cp, "w"))
ror2oa = {v["ror"]: k for k, v in C.items() if v and v.get("ror")}
need = sorted({v["ror"] for v in R.values() if v["ror"] and v["ror"] not in ror2oa})
for k in range(0, len(need), 50):
    q = {"filter": "ror:" + "|".join(sid(x) for x in need[k:k + 50]), "per-page": 50, "select": "id,ror", "api_key": os.environ["OPENALEX_API_KEY"]}
    for r in json.load(urllib.request.urlopen("https://api.openalex.org/institutions?" + urllib.parse.urlencode(q), timeout=60))["results"]:
        ror2oa[r["ror"]] = sid(r["id"])
print(f"no-ID affiliation strings {len(texts)} ({sum(texts.values())} occurrences); ROR chosen {sum(1 for t in texts if R[t]['ror'])}; "
      f"with an OpenAlex institution {sum(1 for t in texts if R[t]['ror'] in ror2oa)}")
for t in texts:
    if R[t]["ror"]:
        print(f"  ROR {R[t]['score']} | {R[t]['name'][:45]:45s} | {t[:100]}")
# (2) in-corpus name match
norm = lambda s: " " + re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()).strip() + " "
GEN = set("of the and for in at de del la le des di du da do university universidad universite universita universitat college hospital "
          "hospitals institute institut instituto istituto center centre centro research medical medicine health sciences science school "
          "national state department faculty laboratory clinic clinical foundation society group company inc ltd gmbh co technology people "
          "s central city general first second third affiliated".split())
CTRY = {"CN": ["china"], "US": ["usa", "united states", "u s a"], "GB": ["uk", "united kingdom", "england", "scotland"], "DE": ["germany"],
        "FR": ["france"], "IT": ["italy"], "ES": ["spain"], "IN": ["india"], "ZA": ["south africa"], "AU": ["australia"], "CA": ["canada", "canad"],
        "NL": ["netherlands"], "KR": ["korea"], "JP": ["japan"], "CL": ["chile"], "BE": ["belgium"], "CZ": ["czechia", "czech"], "FI": ["finland"],
        "CH": ["switzerland"]}
names = collections.defaultdict(set)
for k, v in C.items():
    for n in [v["display_name"]] + (v.get("display_name_alternatives") or []) if v else []:
        toks = norm(n).split()
        if len(toks) >= 2 and not all(x in GEN for x in toks):
            names[norm(n)].add(k)
uniq = 0
for t in texts:
    n = norm(t); hits = {(nn, k) for nn, ks in names.items() if nn in n for k in ks}
    if hits:
        L = max(len(nn) for nn, _ in hits); hits = {(nn, k) for nn, k in hits if len(nn) == L}
    ks = {k for _, k in hits}
    cs = {c for c, ws in CTRY.items() if any(" " + w + " " in n for w in ws)}
    if cs:
        ks = {k for k in ks if C[k].get("country_code") in cs}
    if len(ks) == 1:
        uniq += texts[t]; print(f"  CORPUS | {C[next(iter(ks))]['display_name'][:45]:45s} | {t[:100]}")
print(f"in-corpus unique matches: {uniq} occurrences")
