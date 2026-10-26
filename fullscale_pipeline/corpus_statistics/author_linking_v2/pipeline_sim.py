# The author-identity rules in order, on the working corpus v2 (read-only simulation; nothing is written to the corpus):
# 1d PubMed ORCID preferred (in links_v2.py) -> B (R3: faulty ORCID labels removed) -> A (R2: bylines under untrusted
# OpenAlex IDs removed) -> R1 (merge through shared ORCID) -> R4 strict linking. Also A without B first, and B as deletion.
import itertools, sys
exec(open("orcid_names2.py").read().split("o2b = collections.defaultdict(list)")[0])
ORDER = sys.argv[1] if len(sys.argv) > 1 else "B_then_A"
SRC = sys.argv[2] if len(sys.argv) > 2 else "both"   # which ORCIDs define problem A: "pubmed" (as decided) or "both"
art_n = len({b["pmcid"] for b in B})
log = [("bylines at start", len(B), art_n)]
for b in B: b["orc"] = b["pm_orcid"] or b["oa_orcid"]          # 1d already applied: oa_orcid cleared where it differed
def step_B(bs_all):
    o2b = collections.defaultdict(list)
    for b in bs_all:
        if b["orc"]: o2b[b["orc"]].append(b)
    removed = 0
    for o, bs in o2b.items():
        # refinement: an ORCID on 2+ bylines of one article is not trusted on that article
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
def step_A(bs_all):
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
    kept = [b for b in bs_all if b["oa_id"] not in drop_ids]
    return kept, n_multi, len(drop_ids)
if ORDER == "B_then_A":
    r = step_B(B); log.append(("B: bylines whose ORCID label is removed", r, None))
    kept, nm, nd = step_A(B)
elif ORDER == "A_only":
    kept, nm, nd = step_A(B)
else:
    kept, nm, nd = step_A(B)
    r = step_B(kept); log.append(("B (after A): bylines whose ORCID label is removed", r, None))
lost_arts = art_n - len({b["pmcid"] for b in kept})
log.append((f"A: OpenAlex IDs with 2+ ORCIDs {nm}, not deduplicable {nd}; bylines removed", len(B) - len(kept), lost_arts))
# R1: persons = union of OpenAlex ID and ORCID per byline
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for b in kept:
    ks = [k for k in (("A", b["oa_id"]), ("O", b["orc"])) if k[1]]
    for k in ks: find(k)
    for k in ks[1:]: parent[find(k)] = find(ks[0])
ident = [b for b in kept if b["oa_id"] or b["orc"]]
persons = {find(("A", b["oa_id"]) if b["oa_id"] else ("O", b["orc"])) for b in ident}
oa_ids = {b["oa_id"] for b in ident if b["oa_id"]}
merged_by_orcid = len(oa_ids) + len({b["orc"] for b in ident if b["orc"] and not b["oa_id"]}) - len(persons)
log.append(("R1: identified bylines", len(ident), None)); log.append(("    distinct persons", len(persons), None))
log.append(("    OpenAlex IDs joined into another person through a shared ORCID", merged_by_orcid, None))
P = collections.defaultdict(list)
for b in ident: P[find(("A", b["oa_id"]) if b["oa_id"] else ("O", b["orc"]))].append(b)
bad = [p for p, bs in P.items() if len({x["oa_id"] for x in bs if x["oa_id"]}) > 1 and not all(compatible(bs[0], x) for x in bs)]
log.append(("    persons joined across OpenAlex IDs whose names are different people", len(bad), None))
if "--show" in sys.argv:
    for p in bad:
        bs = P[p]
        print("JOIN", sorted({(f"{x['last']}, {x['fore']}", (x['oa_id'] or '-').rsplit('/',1)[-1], x['orc'] or '-', (x['afftext'] or '')[:55]) for x in bs})[:6])
U = [b for b in kept if not (b["oa_id"] or b["orc"])]
log.append(("unidentified bylines before R4", len(U), None))
for name, v, a in log: print(f"{name:95s} {v:>9,}" + (f"   (articles left without any author: {a:,})" if a is not None and "removed" in name else ""))
