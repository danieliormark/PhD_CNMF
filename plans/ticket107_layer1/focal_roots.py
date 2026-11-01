# All focal model atoms (C/focal) in the trial with article counts and their family under model_levels.level().
import collections, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_levels.py")).read().split("units = []")[0])
cnt = collections.defaultdict(set)
for r in recs:
    for p in r["parents"]:
        for c in p["children"]:
            for a in c["atoms"]:
                if a["atom"].endswith("/C/focal"): cnt[a["atom"].split("/")[0]].add(r["pmcid"])
    for c in r["cousins"]:
        for a in c["atoms"]:
            if a["atom"].endswith("/C/focal"): cnt[a["atom"].split("/")[0]].add(r["pmcid"])
fams = collections.Counter()
for k, v in cnt.items(): fams[level(k, "family")] += len(v)
print("roots", len(cnt), "families", len(fams))
for k in sorted(cnt, key=lambda k: -len(cnt[k])): print(len(cnt[k]), k, "->", level(k, "family"))
