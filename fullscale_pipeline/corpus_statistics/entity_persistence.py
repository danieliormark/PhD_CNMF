# Entity persistence between time slices (owner, 2026-10-16; ticket 107 design doc §7): how much of a slice's
# entities, and of its ties, already existed in the previous slice. Decides between an incremental prior and
# independent fits / one joint model. Working corpus only (28,029). Ticket-107 facets, no atoms:
#   Approach 1 (author facet):    child x article, child x parent, article x journal, author x article, author x affiliation
#   Approach 2 (no author facet): child x article, child x parent, article x journal, article x article (shared author),
#                                 article x affiliation (per byline)
# Identities: child = merged child+cousin layer by G3's role-free content_key; parent by content_key; journal by NLM
# journal ID; author = OpenAlex author ID or ORCID (union-find, as in author_coverage_scopes.py; bylines with neither
# left out); affiliation = OpenAlex institution ID (dedup deferred; bylines without an ID left out, share reported).
# Articles never persist. Ties are distinct entity pairs within a slice, unweighted (weighting not decided).
# Year: OpenAlex publication_year and PubMed epub-else-print year, both reported (owner, 2026-10-16).
#   python3 entity_persistence.py      (reads ~2.7 GB of G3 output once; no network)
import collections, csv, itertools, json, sys

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
G3 = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/g3_test_v1.jsonl"
D, OD = FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
OUT = OD + "entity_persistence.json"

WORK = {l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()}
H = json.load(open(D + "headers.json"))
V = json.load(open(D + "openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
PM = json.load(open(OD + "pubmed_authors.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]
PMID = {r["pmcid"]: r["pmid"] for r in csv.DictReader(open(FP + "corpus_statistics/article_types_valid.csv"))}


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


W = {p: work(p) for p in WORK}
YEAR = {"openalex": {p: str(w["year"]) for p, w in W.items() if w and w.get("year")}, "pubmed": {}}
JOURN = {}
for p in WORK:
    rec = PM.get(PMID.get(p) or "")
    if rec:
        y = rec.get("epub_year") or rec.get("pub_year")
        if y:
            YEAR["pubmed"][p] = str(y)
        nlm = (rec.get("journal") or {}).get("nlm_id")
        if nlm:
            JOURN[p] = nlm

# authors (union-find over OpenAlex ID / ORCID) and affiliations (OpenAlex institution IDs per byline)
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

ART_AUTH = collections.defaultdict(set)      # article -> known-identity persons
ART_AFF = collections.defaultdict(set)       # article -> institution IDs (approach 2: every byline)
AUTH_AFF = collections.defaultdict(set)      # article -> (person, institution) (approach 1: known persons only)
aff_cnt = collections.Counter()
for b in B:
    if b["pmcid"] not in WORK:
        continue
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    person = find(ks[0]) if ks else None
    if person:
        ART_AUTH[b["pmcid"]].add(person)
    w = W.get(b["pmcid"])
    insts = []
    if w and b["oa_idx"] is not None and b["oa_idx"] < len(w["authorships"]):
        insts = [i["id"] for i in w["authorships"][b["oa_idx"]].get("institutions") or [] if i.get("id")]
    aff_cnt["bylines"] += 1
    aff_cnt["with_institution_id"] += bool(insts)
    for i in insts:
        ART_AFF[b["pmcid"]].add(i)
        if person:
            AUTH_AFF[b["pmcid"]].add((person, i))

# G3: merged child layer (children + cousins, role-free content key), parents, child x parent ties
ids = collections.defaultdict(dict)
def iid(f, k):
    d = ids[f]
    return d.setdefault(k, len(d))
ART_CHILD, ART_PAR, ART_CP = (collections.defaultdict(set) for _ in range(3))
n = 0
with open(G3) as fh:
    for line in fh:
        r = json.loads(line)
        p = r["pmcid"]
        if p not in WORK:
            continue
        n += 1
        pkey = {}
        for par in r["parents"]:
            pk = iid("parent", par["content_key"]); pkey[par["id"]] = pk
            ART_PAR[p].add(pk)
            for c in par["children"]:
                ck = iid("child", c["content_key"])
                ART_CHILD[p].add(ck); ART_CP[p].add((ck, pk))
        for c in r["cousins"]:
            ck = iid("child", c["content_key"])
            ART_CHILD[p].add(ck)
            for pid in c["parents"]:
                if pid in pkey:
                    ART_CP[p].add((ck, pkey[pid]))
        if n % 200000 == 0:
            print(f"  {n:,} units read", file=sys.stderr, flush=True)
print(f"G3 units in the working corpus: {n:,}; articles with G3 content: {len(ART_CHILD):,}", file=sys.stderr)


def slice_data(arts, approach):
    """Entities per facet and ties per relation for one slice."""
    E = collections.defaultdict(set)
    R = collections.defaultdict(set)
    E["article"] = set(arts)
    for p in arts:
        for ck in ART_CHILD[p]:
            E["child"].add(ck); R["child x article"].add((ck, p))
        E["parent"] |= ART_PAR[p]
        R["child x parent"] |= ART_CP[p]
        if p in JOURN:
            E["journal"].add(JOURN[p]); R["article x journal"].add((p, JOURN[p]))
        if approach == 1:
            for a in ART_AUTH[p]:
                E["author"].add(a); R["author x article"].add((a, p))
            for a, i in AUTH_AFF[p]:
                E["affiliation"].add(i); R["author x affiliation"].add((a, i))
        else:
            for i in ART_AFF[p]:
                E["affiliation"].add(i); R["article x affiliation"].add((p, i))
    if approach == 2:
        by_auth = collections.defaultdict(list)
        for p in arts:
            for a in ART_AUTH[p]:
                by_auth[a].append(p)
        for ps in by_auth.values():
            for x, y in itertools.combinations(sorted(ps), 2):
                R["article x article"].add((x, y))
    return E, R


FACET_OF = {"child x article": ("child", "article"), "child x parent": ("child", "parent"),
            "article x journal": ("article", "journal"), "author x article": ("author", "article"),
            "author x affiliation": ("author", "affiliation"), "article x article": ("article", "article"),
            "article x affiliation": ("article", "affiliation")}


def compare(a1, a2, approach):
    E1, _ = slice_data(a1, approach)
    E2, R2 = slice_data(a2, approach)
    res = {"n_articles": [len(a1), len(a2)], "facets": {}, "relations": {}}
    P = {f: E1[f] & E2[f] for f in E2}
    pool = [0, 0, 0, 0]     # T2 entities, of which persisting; same without articles
    for f in sorted(E2):
        n1, n2, k = len(E1[f]), len(E2[f]), len(P[f])
        res["facets"][f] = dict(t1=n1, t2=n2, persisting=k, share_of_t2=k / n2 if n2 else None,
                                share_of_t1_reappearing=k / n1 if n1 else None)
        pool[0] += n2; pool[1] += k
        if f != "article":
            pool[2] += n2; pool[3] += k
    res["overall_entities"] = dict(share_of_t2=pool[1] / pool[0], share_of_t2_excl_articles=pool[3] / pool[2])
    tot = [0, 0]
    mass = collections.Counter()      # persisting tie-ends by facet (a tie with two persisting ends counts half each)
    for rel, ties in sorted(R2.items()):
        f1, f2 = FACET_OF[rel]
        touch = both = 0
        for x, y in ties:
            p1, p2 = x in P[f1], y in P[f2]
            if p1 or p2:
                touch += 1
                share = 1 / (p1 + p2)
                if p1: mass[f1] += share
                if p2: mass[f2] += share
            both += p1 and p2
        res["relations"][rel] = dict(t2_ties=len(ties), touching_persisting=touch / len(ties) if ties else None,
                                     both_ends_persisting=both / len(ties) if ties else None)
        tot[0] += len(ties); tot[1] += touch
    res["overall_ties_touching_persisting"] = tot[1] / tot[0] if tot[0] else None
    m = sum(mass.values())
    res["persisting_tie_mass_by_facet"] = {f: v / m for f, v in mass.most_common()} if m else {}
    linked = collections.Counter()
    for p in a2:
        hits = {"child": any(c in P["child"] for c in ART_CHILD[p]),
                "journal": JOURN.get(p) in P["journal"]}
        if approach == 1:
            hits["author"] = any(a in P["author"] for a in ART_AUTH[p])
            hits["affiliation"] = any(i in P["affiliation"] for _, i in AUTH_AFF[p])
        else:
            hits["affiliation"] = any(i in P["affiliation"] for i in ART_AFF[p])
        for f, h in hits.items():
            linked[f] += h
        linked["any"] += any(hits.values())
        linked["any_except_journal"] += any(h for f, h in hits.items() if f != "journal")
    res["t2_articles_with_a_persisting_entity"] = {f: v / len(a2) for f, v in linked.items()}
    return res


SLICES = {"consecutive": [([str(y)], [str(y + 1)]) for y in range(2019, 2026)],
          "long": [(["2019", "2020", "2021", "2022"], ["2023", "2024"]), (["2023", "2024"], ["2025", "2026"])]}
out = {"inputs": dict(g3=G3, units=n, working_corpus=len(WORK), affiliation_bylines=aff_cnt), "results": {}}
for ydef, Y in YEAR.items():
    by = collections.defaultdict(set)
    for p, y in Y.items():
        by[y].add(p)
    out["results"][ydef] = {"articles_per_year": {y: len(s) for y, s in sorted(by.items())},
                            "articles_without_year": len(WORK) - len(Y)}
    for kind, pairs in SLICES.items():
        for s1, s2 in pairs:
            a1 = set().union(*(by[y] for y in s1)); a2 = set().union(*(by[y] for y in s2))
            label = f"{s1[0]}{'-' + s1[-1] if len(s1) > 1 else ''} -> {s2[0]}{'-' + s2[-1] if len(s2) > 1 else ''}"
            for ap in (1, 2):
                out["results"][ydef][f"{kind} | {label} | approach {ap}"] = compare(a1, a2, ap)
            print(f"  done {ydef} {label}", file=sys.stderr, flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT, file=sys.stderr)

pc = lambda x: "—" if x is None else f"{100 * x:.1f}%"
for ydef in YEAR:
    print(f"\n## Year = {ydef}")
    for key, r in out["results"][ydef].items():
        if " | " not in key:
            continue
        f = r["facets"]
        print(f"\n### {key}  (articles {r['n_articles'][0]:,} -> {r['n_articles'][1]:,})")
        print("| facet | T1 | T2 | persisting | % of T2 | % of T1 reappearing |\n|---|---|---|---|---|---|")
        for k, v in f.items():
            print(f"| {k} | {v['t1']:,} | {v['t2']:,} | {v['persisting']:,} | {pc(v['share_of_t2'])} | {pc(v['share_of_t1_reappearing'])} |")
        print("| relation | T2 ties | % touching a persisting entity | % both ends persisting |\n|---|---|---|---|")
        for k, v in r["relations"].items():
            print(f"| {k} | {v['t2_ties']:,} | {pc(v['touching_persisting'])} | {pc(v['both_ends_persisting'])} |")
        print(f"overall: entities persisting {pc(r['overall_entities']['share_of_t2'])} "
              f"(excl. articles {pc(r['overall_entities']['share_of_t2_excl_articles'])}); "
              f"ties touching a persisting entity {pc(r['overall_ties_touching_persisting'])}")
        print("persisting tie mass by facet: " + ", ".join(f"{k} {pc(v)}" for k, v in r["persisting_tie_mass_by_facet"].items()))
        print("T2 articles linked to T1 through: " + ", ".join(f"{k} {pc(v)}" for k, v in r["t2_articles_with_a_persisting_entity"].items()))
