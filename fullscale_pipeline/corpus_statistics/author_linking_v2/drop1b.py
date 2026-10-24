exec(open("orcid_names2.py").read().split("o2b = collections.defaultdict(list)")[0])
import itertools, random
oa_by = collections.defaultdict(list)
for b in B:
    if b["oa_id"]: oa_by[b["oa_id"]].append(b)
multi = [a for a, bs in oa_by.items() if len({x["pm_orcid"] for x in bs if x["pm_orcid"]}) > 1]
keep, drop = [], []
for a in multi:
    bs = oa_by[a]
    names_ok = all(compatible(x, y) for x, y in itertools.combinations(bs, 2))
    by_o = collections.defaultdict(list)
    for x in bs: by_o[x["pm_orcid"] or "-"].append(x)
    orcs = [o for o in by_o if o != "-"]
    def linked(g, h):
        return any((x["inst"] & y["inst"]) or ((x["coauth"] - {x["person"]}) & (y["coauth"] - {y["person"]})) for x in g for y in h if x["pmcid"] != y["pmcid"])
    tied = all(linked(by_o[o], by_o[p]) for o, p in itertools.combinations(orcs, 2))
    (keep if names_ok and tied else drop).append(a)
dropped = [b for a in drop for b in oa_by[a]]
dset = {id(b) for b in dropped}
by_art = collections.defaultdict(list)
for b in B: by_art[b["pmcid"]].append(b)
empty = [p for p in {b["pmcid"] for b in dropped} if all(id(x) in dset for x in by_art[p])]
print(f"OpenAlex IDs with 2+ ORCIDs: {len(multi)}; kept as one person: {len(keep)}; dropped: {len(drop)}")
print(f"bylines dropped: {len(dropped)} in {len({b['pmcid'] for b in dropped})} articles; articles left with no author (dropped): {len(empty)}")
for a in keep:
    print("  KEPT", a.rsplit('/',1)[-1], sorted({(f"{x['last']}, {x['fore']}", x['pm_orcid'] or "-", x['afftext'][:45]) for x in oa_by[a]})[:5])
print("  empty articles:", empty)
