exec(open("hide_recover.py").read().split("random.seed(2026)")[0])
import random
U = [b for b in B if not b["person"]]
pers_aff = collections.defaultdict(dict)
for b in B:
    if b["person"]: pers_aff[b["person"]][b["pmcid"]] = b["afftext"][:85]
L = collections.defaultdict(list); C = collections.Counter()
for b in U:
    t, why = strict(b)
    C[why] += 1
    if t:
        n = len(key2P[b["key"]]); tier = "1" if n == 1 else "2-4" if n < 5 else "5+"
        L[tier].append((b, t))
print("strict B on the unidentified bylines:", dict(C), {k: len(v) for k, v in L.items()})
random.seed(4242)
for tier, k in (("1", 10), ("2-4", 12), ("5+", 12)):
    for b, t in random.sample(L[tier], min(k, len(L[tier]))):
        arts = [a for a in P_ev[t] if a != b["pmcid"] and (b["inst"] & P_ev[t][a][0] or (b["coauth"] & P_ev[t][a][1]))]
        co = sorted({P_name.get(x) or "?" for a in arts for x in (b["coauth"] & P_ev[t][a][1]) - SUSPECT_P})[:3]
        print(f"[{tier}] {b['last']}, {b['fore']} ({b['pmcid']}) | {b['afftext'][:85]}")
        print(f"      -> {P_name[t]} ({arts[0]}) | {pers_aff[t].get(arts[0], '')}   co-authors {co}")
