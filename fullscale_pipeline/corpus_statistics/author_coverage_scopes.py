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
    P = {p for p, a in arts.items() if keep(a["type"])}
    bl = [b for b in B if b["pmcid"] in P]
    ident = lambda b: bool(b["oa_id"] or b["pm_orcid"] or b["oa_orcid"])
    per = collections.defaultdict(lambda: [0, 0, 0])
    fl = collections.defaultdict(list)
    for b in bl:
        v = per[b["pmcid"]]; v[0] += 1; v[1] += bool(b["oa_id"]); v[2] += ident(b)
        if b["pos"] in (0, b["n"] - 1):
            fl[b["pmcid"]].append(ident(b))
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
    people = len({find(k) for k in parent})
    n, na = len(bl), len(P)
    unid = sum(not ident(b) for b in bl)
    pct = lambda x, d: round(100 * x / d, 1) if d else None
    return dict(articles=na, articles_openalex_bylines=len({b["pmcid"] for b in bl if b["src"] == "openalex"}),
                bylines=n, pct_openalex_id=pct(sum(bool(b["oa_id"]) for b in bl), n),
                pct_publisher_orcid=pct(sum(bool(b["pm_orcid"]) for b in bl), n),
                pct_identified=pct(n - unid, n), unidentified_bylines=unid,
                pct_articles_all_openalex_id=pct(sum(v[1] == v[0] for v in per.values()), na),
                pct_articles_all_identified=pct(sum(v[2] == v[0] for v in per.values()), na),
                articles_none_identified=sum(v[2] == 0 for v in per.values()),
                pct_articles_first_or_last_unidentified=pct(sum(not all(v) for v in fl.values()), na),
                identified_people=people, upper_bound_people=people + unid)


S = {k: summarise(f) for k, f in SCOPES.items()}
json.dump(S, open(OD + f"author_coverage_scopes{SUFFIX}.json", "w"), indent=1)
rows = list(S["all"].keys())
print("| | " + " | ".join(S) + " |\n|---|" + "---|" * len(S))
for r in rows:
    print(f"| {r} | " + " | ".join(f"{S[k][r]:,}" if isinstance(S[k][r], int) else str(S[k][r]) for k in S) + " |")
