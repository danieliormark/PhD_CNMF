# Problem C (found 2026-10-26 reading the 31 joins): OpenAlex author IDs whose bylines carry names that are different
# people by the name test, whatever their ORCIDs. Read-only.
exec(open("orcid_names2.py").read().split("o2b = collections.defaultdict(list)")[0])
import random
oa_by = collections.defaultdict(list)
for b in B:
    if b["oa_id"]: oa_by[b["oa_id"]].append(b)
C = collections.Counter(); cases = []
for a, bs in oa_by.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    C["OpenAlex IDs"] += 1
    if len(groups) < 2: continue
    groups.sort(key=len, reverse=True)
    C["IDs with 2+ name groups"] += 1; C["bylines in the minority name groups"] += sum(len(g) for g in groups[1:])
    C["bylines under these IDs"] += len(bs)
    same_sur = len({"".join(tk(g[0]["last"])) for g in groups}) == 1
    C["  of which all groups share the surname"] += same_sur
    cases.append((a, same_sur, [(f"{g[0]['last']}, {g[0]['fore']}", len(g), (g[0]["afftext"] or "")[:40]) for g in groups]))
print(dict(C))
random.seed(11)
for c in random.sample(cases, 20): print(" ", c[0].rsplit("/",1)[-1], "same surname" if c[1] else "DIFFERENT SURNAMES", c[2][:3])
# proposed rule for C: the largest name group keeps the OpenAlex ID; a byline of another name group keeps it only if it
# shares an OpenAlex institution or a co-author's OpenAlex ID with a byline of the largest group (on another article);
# otherwise it loses the ID (a tie: every group must be tied to another group, else all lose it)
by_art = collections.defaultdict(list)
for b in B: by_art[b["pmcid"]].append(b)
for b in B: b["co_ids"] = {c["oa_id"] for c in by_art[b["pmcid"]] if c is not b and c["oa_id"]}
ctry = {}
for b in B:
    w = Wk[b["pmcid"]]; a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    cs = sorted({i.get("country") for i in (a or {}).get("institutions", []) if i.get("country")}); b["ctry"] = cs[0] if len(cs) == 1 else ("several" if cs else "none")
R = collections.Counter(); lose = []; kept_ex = []; lose_ex = []
for a, bs in oa_by.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) < 2: continue
    groups.sort(key=len, reverse=True)
    tie = len(groups[0]) == len(groups[1])
    def tied(g, h): return any((x["inst"] & y["inst"]) or ((x["co_ids"] - {a}) & (y["co_ids"] - {a})) for x in g for y in h if x["pmcid"] != y["pmcid"])
    for g in (groups if tie else groups[1:]):
        others = [h for h in groups if h is not g]
        ok = any(tied(g, h) for h in others) if tie else tied(g, groups[0])
        R["minority bylines kept (tied to the main name)" if ok else "minority bylines losing the OpenAlex ID"] += len(g)
        if ok and len(kept_ex) < 6: kept_ex.append((f"{g[0]['last']}, {g[0]['fore']}", "~", f"{groups[0][0]['last']}, {groups[0][0]['fore']}"))
        if not ok:
            lose += g
            if len(lose_ex) < 6: lose_ex.append((f"{g[0]['last']}, {g[0]['fore']}", "!=", f"{groups[0][0]['last']}, {groups[0][0]['fore']}"))
            for b in g: R["  of which with an own ORCID"] += bool(b["pm_orcid"])
print("C RULE", dict(R)); print("  kept, e.g.", kept_ex); print("  lose, e.g.", lose_ex)
c = collections.Counter(b["ctry"] for b in lose); n = len(lose)
print("  who loses the ID:", {k: f"{v/n:.1%}" for k, v in c.most_common(5)})
