# Pairs of child hyperedges (tier level, numbers left out) that are identical except for their named-model atoms,
# by how the two models relate. A rough size of what soft model similarity in W would add.
import collections, itertools, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
hs = set()
for r in recs:
    for c in [c for p in r["parents"] for c in p["children"]] + r["cousins"]:
        k = key(c["atoms"], "tier")
        if k: hs.add(k)
grp = collections.defaultdict(set)
for k in hs:
    m = frozenset(a.split("/")[0] for a in k if a.endswith("/C/focal") and a.split("/")[0] not in GENERIC)
    if m: grp[frozenset(a for a in k if not (a.endswith("/C/focal") and a.split("/")[0] not in GENERIC))].add(m)
def rel(a, b):
    if len(a) != 1 or len(b) != 1: return "several models"
    a, b = next(iter(a)), next(iter(b))
    if level(a, "version") == level(b, "version"): return "same version, different tier"
    if level(a, "family") == level(b, "family"):
        return "bare family vs versioned" if a == level(a, "family") or b == level(b, "family") else "same family, different version"
    return "different families"
c = collections.Counter()
for rest, ms in grp.items():
    for a, b in itertools.combinations(ms, 2): c[rel(a, b)] += 1
print("named-model hyperedges", sum(len(v) for v in grp.values()), "groups with 2+ model variants", sum(1 for v in grp.values() if len(v) > 1))
for k, v in c.most_common(): print(f"  {k}: {v} pairs")
