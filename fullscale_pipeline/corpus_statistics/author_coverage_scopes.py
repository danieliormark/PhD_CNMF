# Author coverage per corpus scope (owner, 2026-10-10), on top of author_coverage.py's outputs:
#   all     = the whole working corpus, 28,029 articles; the 38 without a PMID have no PubMed author
#             list, so their OpenAlex authorships stand in as bylines (marked, few);
#   article = primary type "Research article" (CORPUS_STATISTICS.md §8), PubMed bylines only;
#   article+review = research articles plus narrative and systematic reviews, for comparison.
import collections, csv, json, re, sys

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OD = FP + "diagnostics/author_coverage/"
SUFFIX = "_refresh" if "--openalex" in sys.argv and sys.argv[sys.argv.index("--openalex") + 1] == "refresh" else ""
V = json.load(open(FP + "diagnostics/openalex_authors/openalex_verified.json"))
arts = {a["pmcid"]: a for a in csv.DictReader(open(FP + f"corpus_statistics/author_coverage_by_article{SUFFIX}.csv"))}
B = [json.loads(l) for l in open(OD + f"bylines{SUFFIX}.jsonl")]
for b in B:
    b["src"] = "pubmed"
norm_orcid = lambda s: (m.group(0).upper() if (m := re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]", s or "")) else None)
if SUFFIX:   # the 38 without a PMID: their refreshed OpenAlex record (same work, or found by DOI)
    OW = json.load(open(OD + "openalex_works.json"))
    for p in [p for p, a in arts.items() if a["pubmed"] == "no_pmid"]:
        w = OW["by_id"].get(V[p][1]["id"].rsplit("/", 1)[-1]) if V.get(p) and V[p][1] else None
        V[p] = [V[p][0], dict(authors=[dict(id=x["id"], orcid=x["orcid"], raw=x["raw"]) for x in w["authorships"]])] if w else [None, None]
for p, a in arts.items():
    if a["pubmed"] == "no_pmid" and V.get(p) and V[p][1]:
        au = V[p][1]["authors"]
        for i, x in enumerate(au):
            B.append(dict(pmcid=p, pos=i, n=len(au), pm_orcid=None, oa_orcid=norm_orcid(x.get("orcid")),
                          oa_id=x.get("id"), src="openalex"))

RESEARCH = "Research article (Journal Article tag, no more specific type)"
REVIEWS = {"Review (narrative / other)", "Systematic review / meta-analysis / scoping review"}
SCOPES = {"all": lambda t: True, "article": lambda t: t == RESEARCH,
          "article+review": lambda t: t == RESEARCH or t in REVIEWS}


def summarise(keep):
    """Bylines and papers (table 1), and authors = distinct people (table 2). Bylines are grouped into one
    person where they share an OpenAlex author ID or an ORCID (from PubMed or the OpenAlex profile); a byline
    with neither cannot be grouped and counts as one person (an overcount if such a person has several)."""
    P = {p for p, a in arts.items() if keep(a["type"])}
    bl = [b for b in B if b["pmcid"] in P]
    oa = lambda b: bool(b["oa_id"])
    pmo = lambda b: bool(b["pm_orcid"])
    anyo = lambda b: bool(b["pm_orcid"] or b["oa_orcid"])
    known = lambda b: oa(b) or anyo(b)
    per = collections.defaultdict(lambda: [0, 0, 0])
    for b in bl:
        v = per[b["pmcid"]]; v[0] += 1; v[1] += oa(b); v[2] += known(b)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for b in bl:
        ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
        for k in ks:
            find(k)
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    person = collections.defaultdict(lambda: [False, False, False])     # OpenAlex ID, PubMed ORCID, any ORCID
    nothing = 0
    for b in bl:
        ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
        if not ks:
            nothing += 1; continue
        v = person[find(ks[0])]; v[0] |= oa(b); v[1] |= pmo(b); v[2] |= anyo(b)
    n, na, npl = len(bl), len(P), len(person) + nothing
    pct = lambda x, d: round(100 * x / d, 1) if d else None
    T1 = {"Papers": na,
          "Papers whose bylines come from OpenAlex (no PubMed record)": len({b["pmcid"] for b in bl if b["src"] == "openalex"}),
          "Bylines": n,
          "% bylines with an OpenAlex author ID known": pct(sum(map(oa, bl)), n),
          "% bylines with an ORCID known from PubMed (publisher-supplied)": pct(sum(map(pmo, bl)), n),
          "% bylines with a PubMed author identifier other than ORCID": 0.0,
          "% bylines with an ORCID known from PubMed or the OpenAlex profile": pct(sum(map(anyo, bl)), n),
          "% bylines with both an OpenAlex author ID and an ORCID known": pct(sum(oa(b) and anyo(b) for b in bl), n),
          "% bylines with an OpenAlex author ID or an ORCID known": pct(sum(map(known, bl)), n),
          "Bylines with neither an OpenAlex author ID nor an ORCID known": sum(not known(b) for b in bl),
          "% papers in which every byline has an OpenAlex author ID known": pct(sum(v[1] == v[0] for v in per.values()), na),
          "% papers in which every byline has an OpenAlex author ID or an ORCID known": pct(sum(v[2] == v[0] for v in per.values()), na),
          "Papers in which no byline has an OpenAlex author ID or an ORCID known": sum(v[2] == 0 for v in per.values())}
    T2 = {"Authors (distinct people, estimated)": npl,
          "  of which: people whose OpenAlex author ID or ORCID we know": len(person),
          "  of which: bylines with neither, one person each": nothing,
          "% authors with an OpenAlex author ID known": pct(sum(v[0] for v in person.values()), npl),
          "% authors with an ORCID known from PubMed (publisher-supplied)": pct(sum(v[1] for v in person.values()), npl),
          "% authors with a PubMed author identifier other than ORCID": 0.0,
          "% authors with an ORCID known from PubMed or the OpenAlex profile": pct(sum(v[2] for v in person.values()), npl),
          "% authors with both an OpenAlex author ID and an ORCID known": pct(sum(v[0] and v[2] for v in person.values()), npl),
          "% authors with an OpenAlex author ID or an ORCID known": pct(len(person), npl),
          "% authors with neither": pct(nothing, npl)}
    return dict(bylines_and_papers=T1, authors=T2)


S = {k: summarise(f) for k, f in SCOPES.items()}
json.dump(S, open(OD + f"author_coverage_scopes{SUFFIX}.json", "w"), indent=1)
for t in ("bylines_and_papers", "authors"):
    print(f"\n| {t} | " + " | ".join(S) + " |\n|---|" + "---|" * len(S))
    for r in S["all"][t]:
        print(f"| {r} | " + " | ".join(f"{S[k][t][r]:,}" if isinstance(S[k][t][r], int) else str(S[k][t][r]) for k in S) + " |")
