# Affiliation coverage by unique author (owner, 2026-10-12): does PMC/PubMed/MEDLINE/Entrez, and does
# OpenAlex, state an affiliation for this author on (at least) one of their working-corpus papers -- the
# affiliation stated for that specific publication, never a profile or "current" affiliation. Authors are
# grouped exactly as in author_coverage_scopes.py (shared OpenAlex author ID or ORCID; a byline with
# neither is its own person). Reads the refreshed OpenAlex snapshot (openalex_works.json) and
# bylines_refresh.jsonl's oa_idx to pick the exact authorship instance aligned to each PubMed byline.
import collections, json

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
D = FP + "diagnostics/openalex_authors/"
OD = FP + "diagnostics/author_coverage/"
H = json.load(open(D + "headers.json"))
V = json.load(open(D + "openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


W = {p: work(p) for p in {b["pmcid"] for b in B}}

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


def aff_flags(b):
    w = W[b["pmcid"]]
    pm = b["n_affs"] > 0
    oa = bool(w and b["oa_idx"] is not None and w["authorships"][b["oa_idx"]].get("raw_affiliations"))
    return pm, oa


# person -> [has a PubMed affiliation on >=1 of their papers, has an OpenAlex affiliation on >=1]
person = collections.defaultdict(lambda: [False, False])
nothing = [0, 0, 0, 0]   # bylines with no OpenAlex ID/ORCID (one person each): total, pubmed, openalex, either
for b in B:
    pm, oa = aff_flags(b)
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    if not ks:
        nothing[0] += 1; nothing[1] += pm; nothing[2] += oa; nothing[3] += pm or oa
        continue
    v = person[find(ks[0])]
    v[0] |= pm; v[1] |= oa

n_known, n_nothing = len(person), nothing[0]
n_all = n_known + n_nothing
pct = lambda x, d: round(100 * x / d, 1)
pm_known = sum(v[0] for v in person.values())
oa_known = sum(v[1] for v in person.values())
either_known = sum(v[0] or v[1] for v in person.values())

print(f"Authors with no OpenAlex ID and no ORCID known (one byline, one person each): {n_nothing:,}")
print(f"  of those, with a PubMed affiliation stated: {nothing[1]:,}; OpenAlex: {nothing[2]:,}; either: {nothing[3]:,}\n")

T = {
    "All authors (estimate)": n_all,
    "  % with an affiliation stated by PubMed/MEDLINE/Entrez": pct(pm_known + nothing[1], n_all),
    "  % with an affiliation stated by OpenAlex": pct(oa_known + nothing[2], n_all),
    "  % with an affiliation stated by either": pct(either_known + nothing[3], n_all),
    "Authors with an OpenAlex author ID or an ORCID known": n_known,
    "  % (of those) with an affiliation stated by PubMed/MEDLINE/Entrez": pct(pm_known, n_known),
    "  % (of those) with an affiliation stated by OpenAlex": pct(oa_known, n_known),
    "  % (of those) with an affiliation stated by either": pct(either_known, n_known),
}
print("| | n / % |\n|---|---|")
for k, v in T.items():
    print(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")

json.dump(dict(table=T, authors_no_id_or_orcid=dict(total=n_nothing, pubmed_aff=nothing[1],
          openalex_aff=nothing[2], either=nothing[3])), open(OD + "author_affiliation_coverage.json", "w"), indent=1)
