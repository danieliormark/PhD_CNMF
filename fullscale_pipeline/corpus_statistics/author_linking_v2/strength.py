# Scratch, read-only: (1) ORCIDs seen with 2+ name keys; (2) strength of the step-1 links (owner 2026-10-24).
exec(open("links_v2.py").read().split("U = [b for b in B if not b[\"person\"]]")[0])
import random
U = [b for b in B if not b["person"]]
# (1) ORCID -> name keys of its bylines (PubMed ORCID, the one kept under decision 1d)
o2k = collections.defaultdict(lambda: collections.defaultdict(list))
for b in B:
    o = b["pm_orcid"] or b["oa_orcid"]
    if o: o2k[o][b["key"] or ("?", b["last"])].append(b)
multi = {o: d for o, d in o2k.items() if len(d) > 1}
same_sur = {o: d for o, d in multi.items() if len({k[0] for k in d}) == 1}
print("ORCIDs:", len(o2k), "with 2+ name keys:", len(multi), "of which same surname, different first given name:", len(same_sur), "; different surname:", len(multi) - len(same_sur))
random.seed(5)
for label, pool in (("same surname", same_sur), ("different surname", {o: d for o, d in multi.items() if o not in same_sur})):
    print("  --", label)
    for o in random.sample(list(pool), min(8, len(pool))):
        print("    ", o, [(f"{v[0]['last']}, {v[0]['fore']}", len(v), v[0]["pmcid"]) for v in pool[o].values()])
# (2) step-1 links: evidence pieces and name commonness (distinct identified people with this key)
S = collections.Counter(); rows = []
for b in U:
    if not b["key"]: continue
    cs = {c: e for c in key2P.get(b["key"], ()) for e in [evidence(b, c)] if e}
    if len(cs) != 1: continue
    c, e = next(iter(cs.items()))
    co = {x for a in e for x in a[2]}; inst = {x for a in e for x in a[1]}; arts = len(e)
    n_same = len(key2P[b["key"]])
    strong = (len(co) >= 2) or (co and inst)
    S["links"] += 1; S["2+ distinct co-authors"] += len(co) >= 2; S["co-author and institution"] += bool(co and inst)
    S["one co-author only, no institution"] += len(co) == 1 and not inst; S["institution only"] += not co
    S["strong (2+ co-authors, or co-author + institution)"] += bool(strong)
    S[f"name key held by {'1' if n_same == 1 else '2-4' if n_same < 5 else '5+'} identified people"] += 1
    rows.append((b, c, e, co, inst, n_same, strong))
for k, v in S.items(): print(f"  {k}: {v}")
print("  strong links by name commonness:", collections.Counter(('1' if r[5]==1 else '2-4' if r[5]<5 else '5+', bool(r[6])) for r in rows))
json.dump([dict(pmcid=r[0]["pmcid"], pos=r[0]["pos"], name=f"{r[0]['last']}, {r[0]['fore']}", aff=r[0]["afftext"][:160],
                target=P_name[r[1]], target_id=r[1][1], evidence=r[2][:3], n_coauthors=len(r[3]), n_inst=len(r[4]),
                same_key_people=r[5], strong=bool(r[6])) for r in rows], open("step1_links.json", "w"), ensure_ascii=False, indent=0)
