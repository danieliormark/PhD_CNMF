# Scratch, read-only: the trial linking rule (author_links_trial.py) on the whole working corpus v2, to count and show
# options B (step 1 only) and C (steps 1 + 2); plus the OpenAlex IDs carrying several ORCIDs (decision 1b).
import collections, json, re, unicodedata, random
FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"; OD = FP + "diagnostics/author_coverage/"; D = FP + "diagnostics/openalex_authors/"
H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
V = json.load(open(D + "openalex_verified.json")); V.update(json.load(open(D + "openalex_verified_new.json")))
OW = json.load(open(OD + "openalex_works.json")); PM = json.load(open(OD + "pubmed_authors.json"))
B = [json.loads(l) for l in open(OD + "bylines_v2_refresh.jsonl")]
# RL-202: articles without a PMID take their bylines from the PMC header (../pmc_header_bylines.py), when built
if __import__("os").path.exists(OD + "bylines_v2_pmc_header.jsonl"):
    B += [json.loads(l) for l in open(OD + "bylines_v2_pmc_header.jsonl")]
def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh
Wk = {p: work(p) for p in {b["pmcid"] for b in B}}
iname = {}
for b in B:
    w = Wk[b["pmcid"]]; a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    b["inst"] = {i["id"] for i in (a or {}).get("institutions", []) if i.get("id")}
    for i in (a or {}).get("institutions", []):
        if i.get("id"): iname[i["id"]] = i.get("name")
    pmr = PM.get((H.get(b["pmcid"]) or {}).get("pmid")) or {}
    pers = [x for x in pmr.get("authors", []) if x["valid"] == "Y" and x["last"]]
    b["afftext"] = (pers[b["pos"]]["affs"][0]["text"] if b["pos"] < len(pers) and pers[b["pos"]]["affs"] else "") or \
        b.get("afftext", "")                                       # PMC-header bylines carry their own (RL-202)
    # 1d (owner): where PubMed and OpenAlex ORCIDs differ, PubMed's (publisher's) is used
    if b["pm_orcid"] and b["oa_orcid"] and b["pm_orcid"] != b["oa_orcid"]: b["oa_orcid"] = None
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
def nkey(last, fore):
    f, l = re.findall(r"[a-z]+", norm(fore)), "".join(re.findall(r"[a-z]+", norm(last)))
    return (l, f[0]) if l and f and len(f[0]) > 1 else None
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for b in B:
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    for k in ks: find(k)
    for k in ks[1:]: parent[find(k)] = find(ks[0])
for i, b in enumerate(B):
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    b["person"] = find(ks[0]) if ks else None; b["key"] = nkey(b["last"], b["fore"]); b["i"] = i
by_art = collections.defaultdict(list)
for b in B: by_art[b["pmcid"]].append(b)
for b in B: b["coauth"] = {c["person"] for c in by_art[b["pmcid"]] if c is not b and c["person"]}
P_ev = collections.defaultdict(lambda: collections.defaultdict(lambda: [set(), set()])); key2P = collections.defaultdict(set); P_name = {}
for b in B:
    if b["person"]:
        P_ev[b["person"]][b["pmcid"]][0] |= b["inst"]; P_ev[b["person"]][b["pmcid"]][1] |= b["coauth"] - {b["person"]}
        if b["key"]: key2P[b["key"]].add(b["person"]); P_name[b["person"]] = f"{b['last']}, {b['fore']}"
def evidence(b, c):
    out = []
    for art, (inst, co) in P_ev[c].items():
        if art == b["pmcid"]: continue
        si, sc = b["inst"] & inst, (b["coauth"] - {c}) & co
        if si or sc: out.append((art, [iname.get(x, x) for x in si], [P_name.get(x, x) for x in sc]))
    return out
U = [b for b in B if not b["person"]]
ukey = collections.defaultdict(list)
for x in U:
    if x["key"]: ukey[x["key"]].append(x)
R = collections.Counter(); ex = collections.defaultdict(list); same_key_any = 0
for b in U:
    if not b["key"]: R["no name key (initial only)"] += 1; d = None
    else:
        allc = key2P.get(b["key"], set()); same_key_any += bool(allc)
        cs = {c: e for c in allc for e in [evidence(b, c)] if e}
        if len(cs) == 1:
            d = "B/C step 1: linked to an identified person"; c, e = next(iter(cs.items())); info = (P_name[c], c[1], e[:2], len(allc))
        elif len(cs) > 1: d = "not linked: 2+ corroborated identified candidates"; info = [(P_name[c], c[1].rsplit("/",1)[-1], ev[:1]) for c, ev in list(cs.items())[:3]]
        else:
            mates = [x for x in ukey[b["key"]] if x is not b and x["pmcid"] != b["pmcid"] and (x["inst"] & b["inst"] or x["coauth"] & b["coauth"])]
            if mates: d = "C step 2 only: linked to other unidentified bylines"; info = [(x["pmcid"], [iname.get(i) for i in x["inst"] & b["inst"]], [P_name.get(p) for p in x["coauth"] & b["coauth"]]) for x in mates[:2]]
            elif allc: d = "not linked: same-name identified people exist, none corroborated"; info = [P_name[c] for c in list(allc)[:3]]
            else: d = "not linked: no candidate (stays own person)"; info = None
        R[d] += 1
        if d and len(ex[d]) < 200: ex[d].append((b["pmcid"], f"{b['last']}, {b['fore']}", [iname.get(i) for i in b["inst"]], b["afftext"][:120], info))
print("unidentified bylines:", len(U)); [print(f"  {k}: {v:,}") for k, v in R.most_common()]
random.seed(24)
for k, v in ex.items():
    print("=====", k)
    for x in random.sample(v, min(4, len(v))): print("  ", json.dumps(x, ensure_ascii=False)[:600])
# 1b: OpenAlex IDs with several ORCIDs (PubMed ORCIDs on bylines carrying that OpenAlex ID)
oa2orc = collections.defaultdict(set); oa_by = collections.defaultdict(list)
for b in B:
    if b["oa_id"]:
        oa_by[b["oa_id"]].append(b)
        if b["pm_orcid"]: oa2orc[b["oa_id"]].add(b["pm_orcid"])
multi = [a for a, o in oa2orc.items() if len(o) > 1]
arts = {b["pmcid"] for a in multi for b in oa_by[a]}
only = [p for p in arts if all(c["oa_id"] in multi for c in by_art[p])]
print("===== 1b: OpenAlex IDs with 2+ PubMed ORCIDs:", len(multi), "; their bylines:", sum(len(oa_by[a]) for a in multi), "; articles:", len(arts), "; articles where they are the only author(s):", len(only))
random.seed(1)
for a in random.sample(multi, 20):
    rows = sorted(((b["pmcid"], b["last"], b["fore"], b["pm_orcid"], (b["afftext"] or "")[:70], (Wk[b["pmcid"]] or {}).get("year")) for b in oa_by[a]), key=lambda r: (r[3] or "", r[5] or 0))
    print(" ", a.rsplit("/",1)[-1]); [print("     ", r) for r in rows[:6]]
# step-2 corroboration kind, and step-1 kind, for the owner's question on B vs C
K = collections.Counter()
for b in U:
    if not b["key"]: continue
    cs = {c: e for c in key2P.get(b["key"], ()) for e in [evidence(b, c)] if e}
    if len(cs) == 1:
        e = next(iter(cs.values())); K["step 1, by co-author"] += any(x[2] for x in e); K["step 1, by institution only"] += not any(x[2] for x in e)
    elif not cs:
        mates = [x for x in ukey[b["key"]] if x is not b and x["pmcid"] != b["pmcid"] and (x["inst"] & b["inst"] or x["coauth"] & b["coauth"])]
        if mates:
            co = any(x["coauth"] & b["coauth"] for x in mates); K["step 2, by co-author"] += co; K["step 2, by institution only"] += not co
print("CORROBORATION", dict(K))
