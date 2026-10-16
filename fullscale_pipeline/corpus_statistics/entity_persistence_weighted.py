# Persistence of child hyperedges with distinctiveness weighting (owner, 2026-10-16; follows entity_persistence.py).
# The unweighted measure counts every child x article tie once, so continuity is dominated by generic phrasing
# ("(dummy_sibling be/P/en)"). Here each T2 tie is weighted as in the toy builder's anchors, log1p(tf) * idf, with
# idf = ln(N / (df + 1)) + 1 computed on the T2 slice alone (N = T2 articles, df = T2 articles using the hyperedge;
# per slice, ticket 106). Variants: unweighted; idf-weighted; multi-word only (>= 2 atoms, no focal-only hyperedge).
# Reports, per slice pair: share of T2 tie mass on persisting hyperedges; share of T2 hyperedges persisting; and the
# distribution over T2 articles of the share of each article's own mass that is persisting. Year: OpenAlex.
import collections, json, math, statistics, sys

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
G3 = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/g3_test_v1.jsonl"
D, OD = FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
WORK = {l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()}
H, V = json.load(open(D + "headers.json")), json.load(open(D + "openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


YEAR = {}
for p in WORK:
    w = work(p)
    if w and w.get("year"):
        YEAR[p] = str(w["year"])

ids, multi = {}, []
TF = collections.defaultdict(collections.Counter)      # article -> child id -> occurrences (tf)
for line in open(G3):
    r = json.loads(line)
    p = r["pmcid"]
    if p not in WORK or p not in YEAR:
        continue
    for c in [c for par in r["parents"] for c in par["children"]] + r["cousins"]:
        k = c["content_key"]
        if k not in ids:
            ids[k] = len(ids)
            at = [a["atom"] for a in c["atoms"]]
            multi.append(len(at) >= 2 and not all(a.endswith("/focal") for a in at))
        TF[p][ids[k]] += 1
print(f"articles {len(TF):,}, child hyperedges {len(ids):,}", file=sys.stderr)

by = collections.defaultdict(set)
for p in TF:
    by[YEAR[p]].add(p)


def compare(s1, s2):
    a1 = set().union(*(by[y] for y in s1)); a2 = set().union(*(by[y] for y in s2))
    E1 = set().union(*(TF[p].keys() for p in a1))
    df = collections.Counter(k for p in a2 for k in TF[p])
    N = len(a2)
    idf = {k: math.log(N / (d + 1)) + 1 for k, d in df.items()}
    res = {"n_articles": [len(a1), len(a2)]}
    for name, wfun, keep in (("unweighted", lambda k, t: 1.0, lambda k: True),
                             ("idf-weighted", lambda k, t: math.log1p(t) * idf[k], lambda k: True),
                             ("multi-word only", lambda k, t: 1.0, lambda k: multi[k]),
                             ("multi-word, idf-weighted", lambda k, t: math.log1p(t) * idf[k], lambda k: multi[k])):
        tot = per = 0.0
        art_share = []
        ents = [k for k in df if keep(k)]
        for p in a2:
            m = mp = 0.0
            for k, t in TF[p].items():
                if not keep(k):
                    continue
                w = wfun(k, t); m += w
                if k in E1:
                    mp += w
            tot += m; per += mp
            if m:
                art_share.append(mp / m)
        q = statistics.quantiles(art_share, n=10) if len(art_share) > 1 else [float("nan")] * 9
        res[name] = dict(tie_mass_persisting=per / tot if tot else None,
                         entities_persisting=sum(k in E1 for k in ents) / len(ents) if ents else None,
                         article_share_p10=q[0], article_share_median=statistics.median(art_share), article_share_p90=q[8],
                         articles_over_10pct=sum(s > 0.10 for s in art_share) / len(a2),
                         articles_over_25pct=sum(s > 0.25 for s in art_share) / len(a2))
    return res


PAIRS = [([str(y)], [str(y + 1)]) for y in range(2019, 2026)] + \
        [(["2019", "2020", "2021", "2022"], ["2023", "2024"]), (["2023", "2024"], ["2025", "2026"])]
out = {}
for s1, s2 in PAIRS:
    lab = f"{s1[0]}{'-' + s1[-1] if len(s1) > 1 else ''} -> {s2[0]}{'-' + s2[-1] if len(s2) > 1 else ''}"
    out[lab] = compare(s1, s2)
json.dump(out, open(OD + "entity_persistence_weighted.json", "w"), indent=1)

pc = lambda x: "—" if x is None else f"{100 * x:.0f}%"
print("| slice pair | variant | % of T2 tie mass persisting | % of T2 hyperedges persisting | per-article share: p10 / median / p90 | articles >10% / >25% |")
print("|---|---|---|---|---|---|")
for lab, r in out.items():
    for v in ("unweighted", "idf-weighted", "multi-word only", "multi-word, idf-weighted"):
        x = r[v]
        print(f"| {lab} | {v} | {pc(x['tie_mass_persisting'])} | {pc(x['entities_persisting'])} | "
              f"{pc(x['article_share_p10'])} / {pc(x['article_share_median'])} / {pc(x['article_share_p90'])} | "
              f"{pc(x['articles_over_10pct'])} / {pc(x['articles_over_25pct'])} |")
