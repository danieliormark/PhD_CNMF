# RL-200 (owner 2026-10-26): the M1 author table — the decided author-identity rules applied to data, in the decided order
# (AUTHOR_IDENTITY.md §8): 1d PubMed ORCID preferred -> R3 (faulty ORCID labels removed) -> R2 (bylines under OpenAlex IDs
# with 2+ PubMed ORCIDs that cannot be deduplicated removed) -> R6 / D1 (bylines of every name other than the most frequent
# under an OpenAlex ID with 2+ names removed; on a tie all) -> R1 (persons = union of OpenAlex ID and ORCID) -> R4 (strict
# linking of bylines without identifier). R5 (institutions to organisations) is applied per subset by
# ../institution_organisations.py. Applied to the whole working corpus v2, since identity evidence (co-authors,
# institutions on other articles) comes from all of it; R2's exception (b) and R4's evidence use raw OpenAlex institution
# IDs, as when the rules were verified. The step code is pipeline_sim.py's, unchanged (step_B, step_A, D1); R4 is
# hide_recover.strict with the persons of the cleaned bylines, so removed bylines cannot serve as co-authors (rule (e)).
# Run from this folder:  python3 author_table.py
# Writes (untracked, holds names): diagnostics/author_coverage/author_table_v2.jsonl, one row per byline: pmcid, pos,
# status (kept / removed_R2 / removed_R6), person (stable key), how (openalex_or_orcid / R4_linked / own_person),
# oa_id, orcid, institutions (raw OpenAlex IDs). Tracked: author_table_v2_summary.json (counts, no names).
import itertools, json, sys
exec(open("orcid_names2.py").read().split("o2b = collections.defaultdict(list)")[0])
SRC = "pubmed"
for b in B: b["orc"] = b["pm_orcid"] or b["oa_orcid"]          # 1d applied in links_v2.py (oa_orcid cleared where it differed)
S = {"bylines at start": len(B), "articles with bylines": len({b["pmcid"] for b in B})}


def step_B(bs_all):                                              # R3 (pipeline_sim.py, unchanged)
    o2b = collections.defaultdict(list)
    for b in bs_all:
        if b["orc"]: o2b[b["orc"]].append(b)
    removed = 0
    for o, bs in o2b.items():
        per_art = collections.Counter(b["pmcid"] for b in bs)
        for b in bs:
            if per_art[b["pmcid"]] > 1: b["orc"] = None; removed += 1
        bs = [b for b in bs if b["orc"]]
        groups = []
        for b in bs:
            for g in groups:
                if compatible(g[0], b): g.append(b); break
            else: groups.append([b])
        if len(groups) < 2: continue
        groups.sort(key=len, reverse=True); ids = [{x["oa_id"] for x in g if x["oa_id"]} for g in groups]
        if all(ids) and set.intersection(*ids): continue
        for g in (groups[1:] if len(groups[0]) > len(groups[1]) else groups):
            for b in g: b["orc"] = None; removed += 1
    return removed


def step_A(bs_all):                                              # R2 (pipeline_sim.py, unchanged)
    oa_by = collections.defaultdict(list)
    for b in bs_all:
        if b["oa_id"]: oa_by[b["oa_id"]].append(b)
    by_art = collections.defaultdict(list)
    for b in bs_all: by_art[b["pmcid"]].append(b)
    for b in bs_all: b["co_ids"] = {c["oa_id"] for c in by_art[b["pmcid"]] if c is not b and c["oa_id"]}
    def linked(g, h):
        return any((x["inst"] & y["inst"]) or ((x["co_ids"] - {x["oa_id"]}) & (y["co_ids"] - {y["oa_id"]})) for x in g for y in h if x["pmcid"] != y["pmcid"])
    drop_ids = set(); n_multi = 0
    for a, bs in oa_by.items():
        aorc = lambda x: (x["orc"] if x["orc"] == x["pm_orcid"] else None) if SRC == "pubmed" else x["orc"]
        orcs = {aorc(x) for x in bs if aorc(x)}
        if len(orcs) < 2: continue
        n_multi += 1
        by_o = collections.defaultdict(list)
        for x in bs:
            if aorc(x): by_o[aorc(x)].append(x)
        ok = all(compatible(x, y) for x, y in itertools.combinations(bs, 2)) and all(linked(by_o[o], by_o[p]) for o, p in itertools.combinations(by_o, 2))
        if not ok: drop_ids.add(a)
    return drop_ids, n_multi


S["R3: bylines whose ORCID label is removed"] = step_B(B)
drop_ids, n_multi = step_A(B)
for b in B:
    b["status"] = "removed_R2" if b["oa_id"] in drop_ids else "kept"
S["R2: OpenAlex IDs with 2+ PubMed ORCIDs"] = n_multi; S["R2: OpenAlex IDs not deduplicable"] = len(drop_ids)
S["R2: bylines removed"] = sum(b["status"] == "removed_R2" for b in B)
kept = [b for b in B if b["status"] == "kept"]
oa_by = collections.defaultdict(list)                            # R6 / D1 (pipeline_sim.py --C, unchanged)
for b in kept:
    if b["oa_id"]: oa_by[b["oa_id"]].append(b)
nC = 0
for a, bs in oa_by.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) < 2: continue
    nC += 1; groups.sort(key=len, reverse=True)
    for g in (groups if len(groups[0]) == len(groups[1]) else groups[1:]):
        for b in g: b["status"] = "removed_R6"
S["R6: OpenAlex IDs with 2+ names"] = nC; S["R6: bylines removed"] = sum(b["status"] == "removed_R6" for b in B)
kept = [b for b in B if b["status"] == "kept"]
arts_all, arts_kept = {b["pmcid"] for b in B}, {b["pmcid"] for b in kept}
S["articles left with no author"] = sorted(arts_all - arts_kept)

parent.clear()                                                   # R1 on the cleaned bylines
for b in kept:
    ks = [k for k in (("A", b["oa_id"]), ("O", b["orc"])) if k[1]]
    for k in ks: find(k)
    for k in ks[1:]: parent[find(k)] = find(ks[0])
for b in B:
    b["person"] = None
for b in kept:
    if b["oa_id"] or b["orc"]:
        b["person"] = find(("A", b["oa_id"]) if b["oa_id"] else ("O", b["orc"])); b["how"] = "openalex_or_orcid"
ident = [b for b in kept if b["person"]]
S["R1: identified bylines"] = len(ident); S["R1: distinct persons"] = len({b["person"] for b in ident})

# R4: evidence from the cleaned persons (hide_recover.strict, with key2P / P_ev rebuilt on the kept bylines)
by_art = collections.defaultdict(list)
for b in kept: by_art[b["pmcid"]].append(b)
for b in kept: b["coauth"] = {c["person"] for c in by_art[b["pmcid"]] if c is not b and c["person"]}
P_ev = collections.defaultdict(lambda: collections.defaultdict(lambda: [set(), set()])); key2P = collections.defaultdict(set)
for b in ident:
    P_ev[b["person"]][b["pmcid"]][0] |= b["inst"]; P_ev[b["person"]][b["pmcid"]][1] |= b["coauth"] - {b["person"]}
    if b["key"]: key2P[b["key"]].add(b["person"])


def strict(b):
    if not b["key"]: return None, "no name key (initial only)"
    cands = key2P.get(b["key"], set()); n = len(cands); ok = {}
    for c in cands:
        co, inst = set(), set()
        for art, (ins, cos) in P_ev[c].items():
            if art == b["pmcid"]: continue
            inst |= b["inst"] & ins; co |= (b["coauth"] - {c}) & cos
        good = (len(co) >= 2 and bool(inst)) if n >= 5 else (len(co) >= 2 or (len(co) >= 1 and bool(inst)))
        if good: ok[c] = 1
    if len(ok) == 1: return next(iter(ok)), "linked"
    return None, "ambiguous: 2+ corroborated candidates" if ok else "no corroborated candidate"


U = [b for b in kept if not b["person"]]
R4 = collections.Counter()
for b in U:
    t, why = strict(b); R4[why] += 1
    if t: b["person"], b["how"] = t, "R4_linked"
for b in U:
    if not b["person"]:
        b["person"], b["how"] = ("U", f"{b['pmcid']}:{b['pos']}"), "own_person"
S["R4: bylines without identifier"] = len(U); S["R4: outcomes"] = dict(R4)
S["persons in the author facet (incl. own persons)"] = len({b["person"] for b in kept})

OUT = OD + "author_table_v2.jsonl"
with open(OUT, "w", encoding="utf-8") as f:
    for b in B:
        f.write(json.dumps(dict(pmcid=b["pmcid"], pos=b["pos"], status=b["status"],
                                person=":".join(b["person"]) if b["person"] else None, how=b.get("how") if b["person"] else None,
                                oa_id=b["oa_id"], orcid=b["orc"], institutions=sorted(b["inst"])), ensure_ascii=False) + "\n")
json.dump(S, open(FP + "corpus_statistics/author_linking_v2/author_table_v2_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k != "articles left with no author"}, indent=1),
      "\narticles left with no author:", S["articles left with no author"])
