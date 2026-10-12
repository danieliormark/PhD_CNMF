# Can unidentified bylines be linked to people by name? (owner question, 2026-10-12; CORPUS_STATISTICS.md §9.6)
# Tested where the truth is known: each identified byline (OpenAlex author ID or ORCID, merged into one
# person as in author_coverage.py) has its identity hidden and is linked back by name, two rules:
#   name only      - link if exactly one other person in the corpus has the same name key;
#   corroborated   - among people with the same name key, link only if exactly one of them shares, on another
#                    article, an OpenAlex institution or an identified co-author with the hidden byline.
# Name key: surname + first given name, accents and case removed (a bare initial gives no key).
# Two populations: people on 2+ corpus articles (a true counterpart exists; a link is right or wrong) and
# people on 1 article (no counterpart; every link is a false merge). Then the rules are applied to the
# real unidentified bylines, against identified people and against each other. Reads the refresh outputs.
import collections, csv, json, re, unicodedata

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OD = FP + "diagnostics/author_coverage/"
H = json.load(open(FP + "diagnostics/openalex_authors/headers.json"))
V = json.load(open(FP + "diagnostics/openalex_authors/openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]

# the OpenAlex work behind each article, as chosen in author_coverage.py --openalex refresh
def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh
W = {p: work(p) for p in {b["pmcid"] for b in B}}
for b in B:
    w = W[b["pmcid"]]
    a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    b["inst"] = {i["id"] for i in (a or {}).get("institutions", []) if i.get("id")}

def nkey(last, fore):
    n = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
    f, l = re.findall(r"[a-z]+", n(fore)), "".join(re.findall(r"[a-z]+", n(last)))
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
    b["key"] = nkey(b["last"], b["fore"])
    b["i"] = i

by_art = collections.defaultdict(list)
for b in B:
    by_art[b["pmcid"]].append(b)
for b in B:
    b["coauth"] = {c["person"] for c in by_art[b["pmcid"]] if c is not b and c["person"]}

# per person: name keys, articles, and evidence per article
P_keys = collections.defaultdict(set); P_arts = collections.defaultdict(set)
P_ev = collections.defaultdict(lambda: collections.defaultdict(lambda: [set(), set()]))
for b in B:
    if b["person"]:
        p = b["person"]
        if b["key"]:
            P_keys[p].add(b["key"])
        P_arts[p].add(b["pmcid"])
        P_ev[p][b["pmcid"]][0] |= b["inst"]; P_ev[p][b["pmcid"]][1] |= b["coauth"] - {p}
key2P = collections.defaultdict(set)
for p, ks in P_keys.items():
    for k in ks:
        key2P[k].add(p)

def corroborated(b, cand, exclude_art):
    for art, (inst, co) in P_ev[cand].items():
        if art == exclude_art:
            continue
        if b["inst"] & inst or (b["coauth"] - {cand}) & co:
            return True
    return False

def decide(b, cands, exclude_art):
    name_only = next(iter(cands)) if len(cands) == 1 else ("ambiguous" if cands else None)
    corr = [c for c in cands if corroborated(b, c, exclude_art)]
    return name_only, (corr[0] if len(corr) == 1 else ("ambiguous" if corr else None))

# ---- held-out test on identified bylines ----
R = {pop: {rule: collections.Counter() for rule in ("name_only", "corroborated")} for pop in ("multi", "single")}
for b in B:
    p = b["person"]
    if not p or not b["key"]:
        continue
    multi = len(P_arts[p]) > 1
    # candidates: people with this name key, with the hidden byline's own article removed from their record
    # p itself is a candidate only if it appears under the same key on another article
    cands = set(key2P[b["key"]]) - {p}
    if multi and any(x["key"] == b["key"] for art in P_arts[p] - {b["pmcid"]} for x in by_art[art] if x["person"] == p):
        cands.add(p)
    pop = "multi" if multi else "single"
    for rule, got in zip(("name_only", "corroborated"), decide(b, cands, b["pmcid"])):
        if got is None:
            out = "no link"
        elif got == "ambiguous":
            out = "ambiguous, no link"
        elif got == p:
            out = "correct link"
        else:
            out = "WRONG link (false merge)"
        R[pop][rule][out] += 1

# ---- the real unidentified bylines ----
U = [b for b in B if not b["person"]]
real = {"to identified people": {"name_only": collections.Counter(), "corroborated": collections.Counter()}}
for b in U:
    cands = set(key2P.get(b["key"], ())) if b["key"] else set()
    no, co = decide(b, cands, b["pmcid"])
    for rule, got in (("name_only", no), ("corroborated", co)):
        real["to identified people"][rule]["no key" if not b["key"] else "no link" if got is None else
                                           "ambiguous" if got == "ambiguous" else "link"] += 1
# placeholder <-> placeholder: same key, different article, sharing an institution or an identified co-author
grp = collections.defaultdict(list)
for b in U:
    if b["key"] and not set(key2P.get(b["key"], ())):
        grp[b["key"]].append(b)
pp = collections.Counter()
for k, bs in grp.items():
    arts = {b["pmcid"] for b in bs}
    if len(arts) < 2:
        continue
    pp["names on 2+ articles"] += 1; pp["their bylines"] += len(bs)
    linked = {b["i"] for x in bs for b in bs if b["pmcid"] != x["pmcid"] and (b["inst"] & x["inst"] or b["coauth"] & x["coauth"])}
    pp["bylines with a corroborated partner"] += len(linked)

# name-key collision among identified people
kc = collections.Counter(len(v) for v in key2P.values())
S = dict(heldout=R, unidentified=real, placeholder_pairs=pp,
         name_keys=dict(keys=len(key2P), keys_with_2plus_people=sum(1 for v in key2P.values() if len(v) > 1),
                        identified_bylines_whose_key_is_shared=sum(1 for b in B if b["person"] and b["key"] and len(key2P[b["key"]]) > 1),
                        identified_bylines_with_key=sum(1 for b in B if b["person"] and b["key"])),
         unidentified_with_any_evidence=dict(total=len(U), with_openalex_institution=sum(bool(b["inst"]) for b in U),
                                             with_identified_coauthor=sum(bool(b["coauth"]) for b in U)))
# The "wrong" corroborated links among single-article people: OpenAlex splits one person into several
# profiles about 1% of the time (§9.3), so some are correct merges OpenAlex missed. Different ORCIDs on the
# two sides show two people (or one person holding two ORCIDs); with ORCIDs from the publisher only, the
# check is independent of OpenAlex. Same ORCIDs cannot occur (they would already be one person).
P_orc = {src: collections.defaultdict(set) for src in ("publisher", "publisher+openalex")}
for b in B:
    if b["person"]:
        for src, os_ in (("publisher", (b["pm_orcid"],)), ("publisher+openalex", (b["pm_orcid"], b["oa_orcid"]))):
            P_orc[src][b["person"]].update(o for o in os_ if o)
wl = {src: collections.Counter() for src in P_orc}
for b in B:
    p = b["person"]
    if not p or not b["key"] or len(P_arts[p]) > 1:
        continue
    corr = [c for c in set(key2P[b["key"]]) - {p} if corroborated(b, c, b["pmcid"])]
    if len(corr) != 1:
        continue
    for src, po in P_orc.items():
        wl[src]["different ORCIDs" if po[p] and po[corr[0]] else "undetermined"] += 1
S["single_article_corroborated_links_by_orcid"] = wl
json.dump(S, open(OD + "name_linking_test.json", "w"), indent=1, default=dict)
print(json.dumps(S, indent=1, default=dict))
