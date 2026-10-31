# Where does a (child, parent) pair recur? Same parent statement in several sentence units, a child twice inside
# one parent, or a cousin meeting the same parent in several sentences.
import collections, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
occ = collections.defaultdict(list); text = {}
for r in recs:
    pk = {p["id"]: p["content_key"] for p in r["parents"]}
    for p in r["parents"]:
        text.setdefault(p["content_key"], (r["pmcid"], r["text"][:150]))
        for c in p["children"]:
            k = key(c["atoms"], "tier")
            if k: occ[(k, p["content_key"])].append(("nested", r["pmcid"], p["id"]))
    for c in r["cousins"]:
        k = key(c["atoms"], "tier")
        for pid in c.get("parents") or []:
            if k and pid in pk: occ[(k, pk[pid])].append(("cousin", r["pmcid"], pid))
rep = {kp: v for kp, v in occ.items() if len(v) > 1}
why = collections.Counter()
for (k, p), v in rep.items():
    ids = {x[2] for x in v}; arts = {x[1] for x in v}
    why["same parent occurrence (child twice inside one parent)" if len(ids) == 1 else
        "parent statement repeated in several sentences, one article" if len(arts) == 1 else
        "parent statement repeated across articles"] += 1
print("pairs", len(occ), "recurring", len(rep)); print(dict(why))
ps = collections.Counter(p for (k, p) in rep)
for p, n in ps.most_common(3): print(n, "recurring pairs in parent:", text[p])
