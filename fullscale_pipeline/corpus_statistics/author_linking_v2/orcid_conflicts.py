# ORCIDs whose bylines carry incompatible names (orcid_names2.py's test): what is left for each minority byline if the
# ORCID stays only with the majority name. Read-only.
exec(open("orcid_names2.py").read().split("o2b = collections.defaultdict(list)")[0])
oa_by = collections.defaultdict(list)
for b in B:
    if b["oa_id"]: oa_by[b["oa_id"]].append(b)
SUSPECT = {a for a, bs in oa_by.items() if len({x["pm_orcid"] for x in bs if x["pm_orcid"]}) > 1}
o2b = collections.defaultdict(list)
for b in B:
    o = b["pm_orcid"] or b["oa_orcid"]
    if o: o2b[o].append(b)
C = collections.Counter(); ex = []
for o, bs in o2b.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) < 2: continue
    groups.sort(key=len, reverse=True); C["ORCIDs"] += 1
    tie = len(groups[0]) == len(groups[1])
    C["tie (no majority name)"] += tie
    maj_oa = {x["oa_id"] for x in groups[0] if x["oa_id"]}
    for g in (groups if tie else groups[1:]):
        for b in g:
            C["bylines losing the ORCID"] += 1
            if not b["oa_id"]: C["  left with no ID"] += 1
            elif b["oa_id"] in maj_oa and not tie: C["  OpenAlex ID = the majority person's ID too"] += 1
            elif b["oa_id"] in SUSPECT: C["  OpenAlex ID is a 1b suspect"] += 1
            else: C["  keeps its own, different OpenAlex ID"] += 1
    if len(ex) < 6 and not tie: ex.append((o, [(f"{g[0]['last']}, {g[0]['fore']}", len(g), sorted({x['oa_id'].rsplit('/',1)[-1] if x['oa_id'] else '-' for x in g})[:2]) for g in groups]))
print(dict(C))
for e in ex: print(" ", e)
# proposed rule: (1) all name groups share one OpenAlex author ID -> one person; (2) else the ORCID stays with the largest
# name group; a tie -> with nobody; (3) bylines that lose the ORCID keep their own OpenAlex ID unless it is a 1b suspect
R = collections.Counter(); exs = collections.defaultdict(list)
for o, bs in o2b.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) < 2: continue
    groups.sort(key=len, reverse=True)
    ids = [{x["oa_id"] for x in g if x["oa_id"]} for g in groups]
    common = set.intersection(*ids) if all(ids) else set()
    if common:
        R["(1) one person: all names share an OpenAlex ID"] += 1; k = "(1)"
        losers = []
    elif len(groups[0]) > len(groups[1]):
        R["(2) ORCID kept by the largest name group"] += 1; k = "(2)"; losers = [b for g in groups[1:] for b in g]
    else:
        R["(2) tie: ORCID kept by nobody"] += 1; k = "tie"; losers = [b for g in groups for b in g]
    for b in losers:
        R["bylines losing the ORCID"] += 1
        R["  -> own OpenAlex ID (clean)" if b["oa_id"] and b["oa_id"] not in SUSPECT else "  -> OpenAlex ID is a 1b suspect (dropped by 1b)" if b["oa_id"] else "  -> no ID left (strict B linking or own person)"] += 1
    if len(exs[k]) < 3: exs[k].append([(f"{g[0]['last']}, {g[0]['fore']}", len(g), sorted({x['oa_id'].rsplit('/',1)[-1] if x['oa_id'] else '-' for x in g})[:2]) for g in groups])
print("PROPOSED", dict(R))
for k, v in exs.items(): print(" ", k, v)
