# Problem C, deletion variants (owner 2026-10-26). D1: delete the bylines of every name group except the largest (on a
# tie, every group). D2: delete every byline under the ID (as R2). Read-only.
exec(open("oa_name_conflicts.py").read().split("# proposed rule for C")[0])
by_art = collections.defaultdict(list)
for b in B: by_art[b["pmcid"]].append(b)
for b in B:
    w = Wk[b["pmcid"]]; a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    cs = sorted({i.get("country") for i in (a or {}).get("institutions", []) if i.get("country")}); b["ctry"] = cs[0] if len(cs) == 1 else ("several" if cs else "none")
D1, D2, big = [], [], []
for a, bs in oa_by.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) < 2: continue
    groups.sort(key=len, reverse=True)
    D2 += bs
    D1 += [b for g in (groups if len(groups[0]) == len(groups[1]) else groups[1:]) for b in g]
    if len(groups[0]) >= 10: big.append((f"{groups[0][0]['last']}, {groups[0][0]['fore']}", len(groups[0]), [(f"{g[0]['last']}, {g[0]['fore']}", len(g)) for g in groups[1:]][:2]))
for name, S in (("D1 delete minority names (ties: all)", D1), ("D2 delete every byline under the ID", D2)):
    ids = {id(b) for b in S}
    arts = {b["pmcid"] for b in S}
    empty = [p for p in arts if all(id(x) in ids for x in by_art[p])]
    c = collections.Counter(b["ctry"] for b in S); n = len(S)
    print(f"{name}: bylines {n}, articles touched {len(arts)}, articles left without author {len(empty)}, China {c['CN']/n:.1%}, US {c['US']/n:.1%}")
print("IDs whose main name has 10+ bylines (D2 would delete all of them):", len(big))
for x in sorted(big, key=lambda x: -x[1])[:8]: print("  ", x)
