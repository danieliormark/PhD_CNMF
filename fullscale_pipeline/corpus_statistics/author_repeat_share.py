# Share of authors with >1 corpus paper, in total and by year (owner, 2026-10-12).
# "Known" (owner's own qualifier): computable only for authors whose OpenAlex author ID or ORCID we
# know (§9.3's grouping) -- without an identity, repeated bylines of the same real person cannot be
# told apart from different people (see the 889 ambiguous name matches in §9.6). A paper's year is the
# OpenAlex work's publication_year (same per-article choice as elsewhere, §9.5); "by year" credits a
# known-identity author to every year in which they have >=1 corpus paper, and asks what share of the
# authors active that year have, in the whole corpus (any year), more than one paper.
import collections, json

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
D = FP + "diagnostics/openalex_authors/"
OD = FP + "diagnostics/author_coverage/"
H = json.load(open(D + "headers.json"))
V = json.load(open(D + "openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]


def work_year(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    ww = by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh
    return str(ww["year"]) if ww and ww.get("year") else None


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

papers = collections.defaultdict(set)   # person -> set of pmcids (known-identity authors only)
for b in B:
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    if ks:
        papers[find(ks[0])].add(b["pmcid"])

n_known = len(papers)
multi = sum(1 for s in papers.values() if len(s) > 1)
print(f"Known-identity authors (OpenAlex ID or ORCID known): {n_known:,}")
print(f"With >1 paper in the corpus, in total: {multi:,} ({100 * multi / n_known:.1f}%)\n")
print("by number of corpus papers (5 = 5 or more):")
for k, v in sorted(collections.Counter(min(len(s), 5) for s in papers.values()).items()):
    print(f"  {k}: {v:,} ({100 * v / n_known:.1f}%)")

year_of = {p: work_year(p) for s in papers.values() for p in s}
by_year = collections.defaultdict(set)
for person, pmcids in papers.items():
    for p in pmcids:
        y = year_of.get(p)
        if y:
            by_year[y].add(person)

print("\n| Year | Known-identity authors active that year | % with >1 corpus paper (any year) |\n|---|---|---|")
for y in sorted(by_year):
    active = by_year[y]
    m = sum(1 for a in active if len(papers[a]) > 1)
    print(f"| {y} | {len(active):,} | {100 * m / len(active):.1f}% |")
