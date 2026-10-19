# Ticket 107 prerequisite P1 (owner, 2026-10-19), trial slices 2020 and 2021 (OpenAlex publication_year).
# Read-only, no network. E: trial article set (research articles + all reviews; borderline types listed).
# F: coverage per slice. C: journals. D (automated layer): G3 invariants and distributions on the trial records.
#   python3 trial_p1_checks.py      (streams the 2.7 GB G3 file once)
import collections, csv, hashlib, json, statistics

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
D, OD = FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
W = {l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()}
H, V = json.load(open(D + "headers.json")), json.load(open(D + "openalex_verified.json"))
OW, PM = json.load(open(OD + "openalex_works.json")), json.load(open(OD + "pubmed_authors.json"))
TYPES = {r["pmcid"]: r for r in csv.DictReader(open(FP + "corpus_statistics/article_types_valid.csv"))}
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]
SLICES = ("2020", "2021")
INCLUDE = {"Research article (Journal Article tag, no more specific type)", "Review (narrative / other)",
           "Systematic review / meta-analysis / scoping review"}


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    bd = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return bd if bd and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


WK = {p: work(p) for p in W}
YEAR = {p: str(w["year"]) for p, w in WK.items() if w and w.get("year")}
ck = lambda atoms: hashlib.sha1(" ".join(sorted(atoms)).encode("utf-8")).hexdigest()[:16]
pc = lambda a, b: f"{100 * a / b:.1f}%" if b else "—"
out = {}

# ---------- E: trial article set ----------
print("## E. Article types in the trial slices (OpenAlex year)\n")
print("| type | 2020 | 2021 | included |\n|---|---|---|---|")
cnt = collections.Counter((TYPES[p]["primary_type"], YEAR[p]) for p in W if YEAR.get(p) in SLICES)
for t in sorted({t for t, _ in cnt}, key=lambda t: -(cnt[(t, "2020")] + cnt[(t, "2021")])):
    print(f"| {t} | {cnt[(t, '2020')]} | {cnt[(t, '2021')]} | {'yes' if t in INCLUDE else 'no'} |")
TRIAL = {y: {p for p in W if YEAR.get(p) == y and TYPES[p]["primary_type"] in INCLUDE} for y in SLICES}
print("\nTrial articles: " + ", ".join(f"{y}: {len(TRIAL[y])}" for y in SLICES))
out["trial_pmcids"] = {y: sorted(s) for y, s in TRIAL.items()}

# ---------- authors / affiliations per article ----------
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
BY = collections.defaultdict(list)
for b in B:
    BY[b["pmcid"]].append(b)

# ---------- D: G3 invariants on the trial records ----------
TRIAL_ALL = TRIAL["2020"] | TRIAL["2021"]
llm = {r["uid"] for r in csv.DictReader(open(PG + "g3_llm_output/llm_output_units.csv"))}
nonprose = {json.loads(l)["uid"] for l in open(PG + "g3_v1/g3_nonprose_v1.jsonl")}
chk = collections.Counter()
size = collections.defaultdict(list); single = collections.Counter(); kinds = collections.Counter()
par_per_art = collections.Counter(); g3_art = set(); examples = collections.defaultdict(list)
for line in open(PG + "g3_v1/g3_test_v1.jsonl"):
    r = json.loads(line)
    p = r["pmcid"]
    if p not in TRIAL_ALL:
        continue
    g3_art.add(p); chk["units"] += 1
    if not r["parents"]:
        chk["unit_without_parent"] += 1; examples["unit_without_parent"].append(r["uid"])
    if r["uid"] in llm:
        chk["llm_output_unit_present"] += 1; examples["llm_output_unit_present"].append(r["uid"])
    if r["uid"] in nonprose:
        chk["nonprose_unit_present"] += 1; examples["nonprose_unit_present"].append(r["uid"])
    pids = set()
    for par in r["parents"]:
        pids.add(par["id"]); par_per_art[p] += 1; chk["parents"] += 1
        keys = [c["content_key"] for c in par["children"]]
        if len(keys) != len(set(keys)):
            chk["duplicate_child_in_parent"] += 1; examples["duplicate_child_in_parent"].append(par["id"])
        if par["content_key"] != ck(keys):
            chk["parent_key_mismatch"] += 1; examples["parent_key_mismatch"].append(par["id"])
        if not any(c["kind"] == "focal_he" for c in par["children"]):
            chk["parent_without_focal_he"] += 1; examples["parent_without_focal_he"].append(par["id"])
        for c in par["children"]:
            at = [a["atom"] for a in c["atoms"]]
            kinds[c["kind"]] += 1; size[c["kind"]].append(len(at)); single[c["kind"]] += len(at) == 1
            if not at:
                chk["empty_hyperedge"] += 1; examples["empty_hyperedge"].append(c["id"])
            if c["content_key"] != ck(at):
                chk["child_key_mismatch"] += 1; examples["child_key_mismatch"].append(c["id"])
            if c["kind"] == "focal_he" and not any(a.endswith("/focal") for a in at):
                chk["focal_he_without_focal_atom"] += 1; examples["focal_he_without_focal_atom"].append(c["id"])
    for c in r["cousins"]:
        at = [a["atom"] for a in c["atoms"]]
        kinds[c["kind"]] += 1; size[c["kind"]].append(len(at)); single[c["kind"]] += len(at) == 1
        if not at:
            chk["empty_hyperedge"] += 1; examples["empty_hyperedge"].append(c["id"])
        if c["content_key"] != ck(at):
            chk["cousin_key_mismatch"] += 1; examples["cousin_key_mismatch"].append(c["id"])
        if not c["parents"] or any(x not in pids for x in c["parents"]):
            chk["cousin_parent_missing"] += 1; examples["cousin_parent_missing"].append(c["id"])
print("\n## D. G3 invariants on the trial records\n")
for k in ("units", "parents", "unit_without_parent", "llm_output_unit_present", "nonprose_unit_present",
          "parent_key_mismatch", "child_key_mismatch", "cousin_key_mismatch", "empty_hyperedge",
          "duplicate_child_in_parent", "parent_without_focal_he", "focal_he_without_focal_atom", "cousin_parent_missing"):
    print(f"  {k:30s} {chk[k]:,}" + (f"   e.g. {examples[k][:3]}" if examples[k] else ""))
print("\nHyperedge size by kind (atoms): count, median, p90, max, % single-word")
for k in sorted(kinds):
    s = sorted(size[k]); q = statistics.quantiles(s, n=10)[8] if len(s) > 1 else s[0]
    print(f"  {k:14s} {kinds[k]:7,}  median {statistics.median(s):.0f}  p90 {q:.0f}  max {s[-1]}  single {pc(single[k], kinds[k])}")
for y in SLICES:
    v = [par_per_art[p] for p in TRIAL[y]]
    q = statistics.quantiles(v, n=10)
    top = sorted(TRIAL[y], key=lambda p: -par_per_art[p])[:3]
    print(f"Parents per article {y}: median {statistics.median(v):.0f}, p10 {q[0]:.0f}, p90 {q[8]:.0f}, max {max(v)}; "
          f"largest: {[(p, par_per_art[p]) for p in top]}")
out["g3_checks"] = dict(counts=chk, examples={k: v[:20] for k, v in examples.items()})

# ---------- F: coverage per slice ----------
print("\n## F. Coverage per trial slice\n")
print("| | 2020 | 2021 |\n|---|---|---|")
rows = collections.OrderedDict()
for y in SLICES:
    A = TRIAL[y]; n = len(A)
    bl = [b for p in A for b in BY[p]]
    known = [b for b in bl if b["oa_id"] or b["pm_orcid"] or b["oa_orcid"]]
    def has_inst(p):
        w = WK.get(p)
        return bool(w) and any(i.get("id") for a in w["authorships"] for i in a.get("institutions") or [])
    def has_known(p):
        return any(b["oa_id"] or b["pm_orcid"] or b["oa_orcid"] for b in BY[p])
    inst_bylines = sum(1 for b in bl if WK.get(b["pmcid"]) and b["oa_idx"] is not None
                       and b["oa_idx"] < len(WK[b["pmcid"]]["authorships"])
                       and any(i.get("id") for i in WK[b["pmcid"]]["authorships"][b["oa_idx"]].get("institutions") or []))
    persons = {find(next(k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1])) for b in known}
    insts = {i["id"] for p in A if WK.get(p) for a in WK[p]["authorships"] for i in a.get("institutions") or [] if i.get("id")}
    jr = {(PM.get(TYPES[p]["pmid"]) or {}).get("journal", {}).get("nlm_id") for p in A} - {None}
    vals = {"articles": n, "with G3 content": sum(p in g3_art for p in A),
            "with a journal (NLM ID)": sum(bool((PM.get(TYPES[p]["pmid"]) or {}).get("journal", {}).get("nlm_id")) for p in A),
            "with >=1 OpenAlex institution ID": sum(has_inst(p) for p in A),
            "with >=1 author whose OpenAlex ID or ORCID we know": sum(has_known(p) for p in A),
            "bylines": len(bl), "bylines whose OpenAlex ID or ORCID we know": len(known),
            "bylines with an OpenAlex institution ID": inst_bylines,
            "distinct authors (known ID)": len(persons), "distinct institutions (OpenAlex ID)": len(insts),
            "distinct journals (NLM ID)": len(jr)}
    rows[y] = vals
for k in rows["2020"]:
    a, b = rows["2020"][k], rows["2021"][k]
    base = "articles" if k.startswith("with") else ("bylines" if k.startswith("bylines ") else None)
    fmt = lambda v, y: f"{v:,}" + (f" ({pc(v, rows[y][base])})" if base else "")
    print(f"| {k} | {fmt(a, '2020')} | {fmt(b, '2021')} |")
out["coverage"] = rows

# ---------- C: journals ----------
print("\n## C. Journals in the trial slices\n")
J = collections.defaultdict(set)
for y in SLICES:
    for p in TRIAL[y]:
        j = (PM.get(TYPES[p]["pmid"]) or {}).get("journal") or {}
        if j.get("nlm_id"):
            J[j["nlm_id"]].add((j.get("title") or "").strip().lower())
norm = collections.defaultdict(set)
for nid, titles in J.items():
    for t in titles:
        norm[t.replace("the ", "").replace("&", "and").strip(" .")].add(nid)
dups = {t: ids for t, ids in norm.items() if len(ids) > 1}
multi_title = {nid: t for nid, t in J.items() if len(t) > 1}
print(f"Distinct NLM IDs: {len(J)}; titles shared by 2+ NLM IDs: {len(dups)} {list(dups.items())[:5]}; "
      f"NLM IDs with 2+ titles: {len(multi_title)} {list(multi_title.items())[:5]}")
missing = [p for y in SLICES for p in TRIAL[y] if not (PM.get(TYPES[p]["pmid"]) or {}).get("journal", {}).get("nlm_id")]
print(f"Trial articles without an NLM journal ID: {len(missing)} {missing[:10]}")
out["journals"] = dict(n=len(J), shared_titles={t: sorted(i) for t, i in dups.items()}, missing=missing)
json.dump(out, open(OD + "trial_p1_checks.json", "w"), indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))
