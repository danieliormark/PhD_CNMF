# Author coverage of the working corpus (working_corpus_pmcids.txt: G3-valid minus retracted,
# CORPUS_STATISTICS.md §8-§9). Pipeline agreed 2026-10-10 (owner):
#   - article-anchored; the byline backbone is PubMed's own AuthorList (pubmed_authors_fetch.py):
#     name, order, ORCID where supplied, group (collective) names;
#   - OpenAlex only as the identity layer: the work already matched per article in
#     corpus_statistics/verify.py (diagnostics/openalex_authors/openalex_verified.json), always the
#     record of the version in the corpus (its DOI = the PMC header DOI), never a repository record;
#   - inside one article, PubMed bylines and OpenAlex authorships are aligned in order, a pair counting
#     only when the surname agrees (no positional guessing).
# Writes corpus_statistics/author_coverage_by_article.csv (no names) and, untracked,
# diagnostics/author_coverage/bylines.jsonl and author_coverage_summary.json.
import collections, csv, json, re, unicodedata

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
D = FP + "diagnostics/openalex_authors/"
OD = FP + "diagnostics/author_coverage/"
wc = [l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()]
H = json.load(open(D + "headers.json"))
V = json.load(open(D + "openalex_verified.json"))
PM = json.load(open(OD + "pubmed_authors.json"))
TYPES = {r["pmcid"]: r["primary_type"] for r in csv.DictReader(open(FP + "corpus_statistics/article_types_valid.csv"))}

SAME_VERSION = {"pmid+doi", "doi", "pmid_only_no_header_doi", "doi_differs_from_pmid_hit"}
OTHER_VERSION = {"pmid_doi_mismatch_no_doi_hit"}   # OpenAlex record is another publication (abstract, chapter)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z]+", s)


def surname_matches(pm, oa_raw):
    """pm = (last, fore). Surname: every PubMed surname token in the OpenAlex name, or the joined surname
    equal to a run of consecutive OpenAlex tokens ("de Freitas"/"DeFreitas"). Forename: when both sides
    have a given-name token left after removing the surname once, some initial must agree."""
    pm_last, pm_fore = pm
    lt, ot = toks(pm_last), toks(oa_raw)
    if not lt or not ot:
        return False
    joined = "".join(lt)
    if not (all(t in ot for t in lt) or any("".join(ot[i:j]) == joined
                                            for i in range(len(ot)) for j in range(i + 1, len(ot) + 1))):
        return False
    rest = list(ot)
    for t in lt:
        if t in rest:
            rest.remove(t)
    ft = toks(pm_fore)
    if not ft or not rest:
        return True
    return any(r[0] == f[0] for r in rest for f in ft)


def align(pm, oa):
    """Order-preserving alignment maximising surname-confirmed pairs (LCS)."""
    n, m = len(pm), len(oa)
    if n * m > 4_000_000:                      # a consortium list; match greedily in order
        pairs, j = {}, 0
        for i in range(n):
            for k in range(j, min(m, j + 50)):
                if surname_matches(pm[i], oa[k]):
                    pairs[i] = k; j = k + 1; break
        return pairs
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for k in range(m - 1, -1, -1):
            L[i][k] = L[i + 1][k + 1] + 1 if surname_matches(pm[i], oa[k]) else max(L[i + 1][k], L[i][k + 1])
    pairs, i, k = {}, 0, 0
    while i < n and k < m:
        if surname_matches(pm[i], oa[k]) and L[i][k] == L[i + 1][k + 1] + 1:
            pairs[i] = k; i += 1; k += 1
        elif L[i + 1][k] >= L[i][k + 1]:
            i += 1
        else:
            k += 1
    return pairs


norm_orcid = lambda s: (m.group(0).upper() if (m := re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]", s or "")) else None)

bylines, arts = [], []
stat = collections.Counter()
for pmcid in wc:
    pmid = (H.get(pmcid) or {}).get("pmid")
    method, work = V.get(pmcid) or ["unmatched", None]
    oa_status = ("same_version" if method in SAME_VERSION and work else
                 "other_version" if method in OTHER_VERSION and work else "none")
    rec = PM.get(pmid) if pmid else None
    if not pmid:
        stat["articles_no_pmid"] += 1
        arts.append(dict(pmcid=pmcid, pmid="", pubmed="no_pmid", oa=oa_status, year="", type=TYPES[pmcid],
                         n_bylines=0, n_group=0, n_oa_aligned=0, n_oa_id=0, n_orcid=0, n_identified=0,
                         n_oa_authorships=len(work["authors"]) if work else 0, complete=""))
        continue
    if rec is None:
        stat["articles_pmid_not_in_pubmed"] += 1
        continue
    personal = [a for a in rec["authors"] if a["valid"] == "Y" and a["last"]]
    group = [a for a in rec["authors"] if a["valid"] == "Y" and a["collective"]]
    oa = work["authors"] if work and oa_status != "none" else []
    pairs = align([(a["last"], a["fore"]) for a in personal], [x["raw"] for x in oa]) if oa else {}
    year = rec["epub_year"] or rec["pub_year"] or ""
    n_id = n_orcid = n_ident = 0
    for i, a in enumerate(personal):
        x = oa[pairs[i]] if i in pairs else None
        pm_orcid = next((norm_orcid(v) for s, v in a["ids"] if (s or "").upper() == "ORCID"), None)
        oa_orcid = norm_orcid(x["orcid"]) if x and x.get("orcid") else None
        oa_id = x["id"] if x and x.get("id") else None
        n_id += bool(oa_id); n_orcid += bool(pm_orcid or oa_orcid); n_ident += bool(oa_id or pm_orcid or oa_orcid)
        bylines.append(dict(pmcid=pmcid, pos=i, n=len(personal), last=a["last"], fore=a["fore"],
                            pm_orcid=pm_orcid, oa_orcid=oa_orcid, oa_id=oa_id, aligned=x is not None,
                            oa_status=oa_status, year=year, n_affs=len(a["affs"])))
    arts.append(dict(pmcid=pmcid, pmid=pmid, pubmed="ok", oa=oa_status, year=year, type=TYPES[pmcid],
                     n_bylines=len(personal), n_group=len(group), n_oa_aligned=len(pairs), n_oa_id=n_id,
                     n_orcid=n_orcid, n_identified=n_ident, n_oa_authorships=len(oa), complete=rec["complete"]))

with open(FP + "corpus_statistics/author_coverage_by_article.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(arts[0].keys()))
    w.writeheader(); w.writerows(arts)
with open(OD + "bylines.jsonl", "w") as f:
    for b in bylines:
        f.write(json.dumps(b, ensure_ascii=False) + "\n")

# ---- summary ----
S = {}
A = [a for a in arts if a["pubmed"] == "ok"]
S["articles"] = dict(working_corpus=len(wc), with_pmid_and_pubmed_record=len(A), **stat,
                     author_list_incomplete=sum(a["complete"] == "N" for a in A),
                     no_personal_byline=sum(a["n_bylines"] == 0 for a in A),
                     with_group_author=sum(a["n_group"] > 0 for a in A),
                     oa=collections.Counter(a["oa"] for a in A))
B = bylines
S["bylines"] = dict(total=len(B), aligned_to_openalex=sum(b["aligned"] for b in B),
                    with_openalex_id=sum(bool(b["oa_id"]) for b in B),
                    with_orcid_pubmed=sum(bool(b["pm_orcid"]) for b in B),
                    with_orcid_any=sum(bool(b["pm_orcid"] or b["oa_orcid"]) for b in B),
                    identified_id_or_orcid=sum(bool(b["oa_id"] or b["pm_orcid"] or b["oa_orcid"]) for b in B),
                    neither=sum(not (b["oa_id"] or b["pm_orcid"] or b["oa_orcid"]) for b in B))
S["alignment"] = dict(
    articles_same_count=sum(a["n_bylines"] == a["n_oa_authorships"] for a in A if a["oa"] != "none"),
    articles_all_bylines_aligned=sum(a["n_oa_aligned"] == a["n_bylines"] for a in A if a["oa"] != "none"),
    unaligned_bylines_with_oa_record=sum(not b["aligned"] for b in B if b["oa_status"] != "none"),
    unaligned_openalex_authorships=sum(a["n_oa_authorships"] - a["n_oa_aligned"] for a in A if a["oa"] != "none"))
pa = [a for a in A if a["n_bylines"]]
S["per_article"] = dict(
    all_bylines_openalex_id=sum(a["n_oa_id"] == a["n_bylines"] for a in pa),
    all_bylines_id_or_orcid=sum(a["n_identified"] == a["n_bylines"] for a in pa),
    at_least_one_openalex_id=sum(a["n_oa_id"] > 0 for a in pa),
    at_least_one_identified=sum(a["n_identified"] > 0 for a in pa),
    none_identified=sum(a["n_identified"] == 0 for a in pa),
    first_or_last_unidentified=0)
fl = collections.defaultdict(dict)
for b in B:
    if b["pos"] in (0, b["n"] - 1):
        fl[b["pmcid"]][b["pos"]] = bool(b["oa_id"] or b["pm_orcid"] or b["oa_orcid"])
S["per_article"]["first_or_last_unidentified"] = sum(not all(v.values()) for v in fl.values())

# distinct people: OpenAlex IDs and ORCIDs merged (union-find); unidentified bylines as bounds
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)
for b in B:
    keys = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    for k in keys:
        find(k)
    for k in keys[1:]:
        union(keys[0], k)
roots = {find(k) for k in parent}
S["distinct_people"] = dict(openalex_ids=len({b["oa_id"] for b in B if b["oa_id"]}),
                            orcids=len({o for b in B for o in (b["pm_orcid"], b["oa_orcid"]) if o}),
                            merged_ids_and_orcids=len(roots),
                            unidentified_bylines=S["bylines"]["neither"],
                            upper_bound=len(roots) + S["bylines"]["neither"])
# disambiguation check against the publisher-supplied ORCID (independent of OpenAlex)
orc2oa, oa2orc = collections.defaultdict(set), collections.defaultdict(set)
for b in B:
    if b["pm_orcid"] and b["oa_id"]:
        orc2oa[b["pm_orcid"]].add(b["oa_id"]); oa2orc[b["oa_id"]].add(b["pm_orcid"])
S["disambiguation_check"] = dict(
    pubmed_orcids_with_openalex_id=len(orc2oa),
    orcid_split_over_several_openalex_ids=sum(len(v) > 1 for v in orc2oa.values()),
    openalex_ids_seen_with_pubmed_orcid=len(oa2orc),
    openalex_id_carrying_several_orcids=sum(len(v) > 1 for v in oa2orc.values()),
    bylines_pubmed_orcid_differs_from_openalex_orcid=sum(
        1 for b in B if b["pm_orcid"] and b["oa_orcid"] and b["pm_orcid"] != b["oa_orcid"]))
by_year = collections.defaultdict(lambda: [0, 0, 0])
for b in B:
    y = b["year"] or "?"
    by_year[y][0] += 1; by_year[y][1] += bool(b["oa_id"]); by_year[y][2] += bool(b["oa_id"] or b["pm_orcid"] or b["oa_orcid"])
S["by_year"] = {y: dict(bylines=v[0], openalex_id=v[1], identified=v[2]) for y, v in sorted(by_year.items())}
by_type = collections.defaultdict(lambda: [0, 0, 0])
tmap = {a["pmcid"]: a["type"] for a in A}
for b in B:
    t = tmap[b["pmcid"]]
    by_type[t][0] += 1; by_type[t][1] += bool(b["oa_id"]); by_type[t][2] += bool(b["oa_id"] or b["pm_orcid"] or b["oa_orcid"])
S["by_type"] = {t: dict(bylines=v[0], openalex_id=v[1], identified=v[2]) for t, v in sorted(by_type.items(), key=lambda x: -x[1][0])}
NP = [a for a in arts if a["pubmed"] == "no_pmid"]
S["no_pmid_tail"] = dict(articles=len(NP), with_openalex_record=sum(a["oa"] != "none" for a in NP),
                         openalex_authorships=sum(a["n_oa_authorships"] for a in NP))
json.dump(S, open(OD + "author_coverage_summary.json", "w"), indent=1, default=dict)
print(json.dumps(S, indent=1, default=dict))
