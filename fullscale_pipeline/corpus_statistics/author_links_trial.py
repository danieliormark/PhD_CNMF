# Ticket 107 P1.A (owner, 2026-10-19): apply the current author-linking method (name_linking_test.py, CORPUS_STATISTICS
# §9.6; checked in name_linking_checks.py) to the bylines of the trial slices that have neither an OpenAlex author ID
# nor an ORCID. Corroborated rule, name key = surname + first given name: (1) link to a person whose ID we know if
# exactly one same-key person shares, on another article, an OpenAlex institution or a co-author whose ID we know;
# (2) otherwise link unlinked bylines with each other under the same corroboration (corpus-wide evidence).
# Output: author_links_trial.csv (one row per trial byline without an ID: decision, target, evidence). No network.
import collections, csv, json, re, unicodedata

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OD = FP + "diagnostics/author_coverage/"
H = json.load(open(FP + "diagnostics/openalex_authors/headers.json"))
V = json.load(open(FP + "diagnostics/openalex_authors/openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]
T = json.load(open(OD + "trial_p1_checks.json"))["trial_pmcids"]
TRIAL = set(T["2020"]) | set(T["2021"]) | {"PMC7815194", "PMC8051288", "PMC8523728", "PMC8666839", "PMC8752034"}


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


Wk = {p: work(p) for p in {b["pmcid"] for b in B}}
iname = {}
for b in B:
    w = Wk[b["pmcid"]]
    a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    b["inst"] = {i["id"] for i in (a or {}).get("institutions", []) if i.get("id")}
    for i in (a or {}).get("institutions", []):
        if i.get("id"):
            iname[i["id"]] = i.get("name")
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
def nkey(last, fore):
    f, l = re.findall(r"[a-z]+", norm(fore)), "".join(re.findall(r"[a-z]+", norm(last)))
    return (l, f[0]) if l and f and len(f[0]) > 1 else None
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for b in B:
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    for k in ks:
        find(k)
    for k in ks[1:]:
        parent[find(k)] = find(ks[0])
for i, b in enumerate(B):
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    b["person"] = find(ks[0]) if ks else None
    b["key"] = nkey(b["last"], b["fore"]); b["i"] = i
by_art = collections.defaultdict(list)
for b in B:
    by_art[b["pmcid"]].append(b)
for b in B:
    b["coauth"] = {c["person"] for c in by_art[b["pmcid"]] if c is not b and c["person"]}
P_ev = collections.defaultdict(lambda: collections.defaultdict(lambda: [set(), set()]))
key2P = collections.defaultdict(set); P_name = {}
for b in B:
    if b["person"]:
        P_ev[b["person"]][b["pmcid"]][0] |= b["inst"]; P_ev[b["person"]][b["pmcid"]][1] |= b["coauth"] - {b["person"]}
        if b["key"]:
            key2P[b["key"]].add(b["person"]); P_name[b["person"]] = f"{b['last']}, {b['fore']}"
def evidence(b, c):
    out = []
    for art, (inst, co) in P_ev[c].items():
        if art == b["pmcid"]:
            continue
        si, sc = b["inst"] & inst, (b["coauth"] - {c}) & co
        if si or sc:
            out.append(f"{art}: " + "; ".join([f"institution {iname.get(x, x)}" for x in si] + [f"co-author {P_name.get(x, x)}" for x in sc]))
    return out
U = [b for b in B if not b["person"]]
rows = []
for b in U:
    if b["pmcid"] not in TRIAL:
        continue
    row = dict(pmcid=b["pmcid"], position=b["pos"], name=f"{b['last']}, {b['fore'] or ''}", decision="", target="", evidence="")
    if not b["key"]:
        row["decision"] = "not linked: no name key (initial only)"
    else:
        cs = {c: evidence(b, c) for c in key2P.get(b["key"], ())}
        cs = {c: e for c, e in cs.items() if e}
        if len(cs) == 1:
            c, e = next(iter(cs.items()))
            row.update(decision="linked to a person whose ID we know", target=f"{P_name[c]} ({c[1]})", evidence=" | ".join(e))
        elif len(cs) > 1:
            row["decision"] = "not linked: ambiguous (2+ corroborated candidates)"
        else:
            mates = [x for x in U if x is not b and x["key"] == b["key"] and x["pmcid"] != b["pmcid"]
                     and (x["inst"] & b["inst"] or x["coauth"] & b["coauth"])]
            if mates:
                row.update(decision="linked to another byline without an ID", target="; ".join(f"{x['pmcid']} #{x['pos']}" for x in mates),
                           evidence=" | ".join(f"{x['pmcid']}: " + "; ".join([f"institution {iname.get(i, i)}" for i in x["inst"] & b["inst"]] +
                                                                         [f"co-author {P_name.get(p, p)}" for p in x["coauth"] & b["coauth"]]) for x in mates))
            else:
                row["decision"] = "not linked: no corroborated candidate (stays its own person)"
    rows.append(row)
with open("author_links_trial.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), collections.Counter(r["decision"] for r in rows))
for r in rows:
    if r["decision"].startswith("linked"):
        print(" ", r["pmcid"], r["name"], "->", r["target"], "|", r["evidence"][:160])
