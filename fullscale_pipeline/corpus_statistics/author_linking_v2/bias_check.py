# Who is affected by R2 (1b drop) and R3 (ORCID removed): country of the byline's OpenAlex institution(s), team size,
# against all bylines. Read-only.
exec(open("orcid_conflicts.py").read().split("C = collections.Counter(); ex = []")[0])
import itertools
ctry = {}
for b in B:
    w = Wk[b["pmcid"]]; a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    cs = sorted({i.get("country") for i in (a or {}).get("institutions", []) if i.get("country")})
    b["ctry"] = cs[0] if len(cs) == 1 else ("several" if cs else "none")
# R2 dropped bylines (drop1b.py's rule)
exec(open("drop1b.py").read().split("multi = ")[1].join(["multi = ", ""]) if False else "")
multi = [a for a, bs in oa_by.items() if len({x["pm_orcid"] for x in bs if x["pm_orcid"]}) > 1]
def linked(g, h):
    return any((x["inst"] & y["inst"]) or ((x["coauth"] - {x["person"]}) & (y["coauth"] - {y["person"]})) for x in g for y in h if x["pmcid"] != y["pmcid"])
R2 = []
for a in multi:
    bs = oa_by[a]; by_o = collections.defaultdict(list)
    for x in bs: by_o[x["pm_orcid"] or "-"].append(x)
    orcs = [o for o in by_o if o != "-"]
    if not (all(compatible(x, y) for x, y in itertools.combinations(bs, 2)) and all(linked(by_o[o], by_o[p]) for o, p in itertools.combinations(orcs, 2))):
        R2 += bs
# R3 losers
R3 = []
for o, bs in o2b.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) < 2: continue
    groups.sort(key=len, reverse=True); ids = [{x["oa_id"] for x in g if x["oa_id"]} for g in groups]
    if all(ids) and set.intersection(*ids): continue
    R3 += [b for g in (groups[1:] if len(groups[0]) > len(groups[1]) else groups) for b in g]
def prof(S):
    c = collections.Counter(b["ctry"] for b in S); n = len(S)
    return n, {k: f"{v/n:.1%}" for k, v in c.most_common(6)}, f"median team {sorted(b['n'] for b in S)[n//2]}"
for name, S in (("all bylines", B), ("R2 dropped (1b)", R2), ("R3 lose the ORCID", R3)):
    print(name, *prof(S))
