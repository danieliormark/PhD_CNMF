exec(open("hide_recover.py").read().split("random.seed(2026)")[0])
import random
arts_of = collections.defaultdict(set)
for b in B:
    if b["person"]: arts_of[b["person"]].add(b["pmcid"])
single = [b for b in B if b["person"] and len(arts_of[b["person"]]) == 1 and b["person"] not in SUSPECT_P and b["key"]]
random.seed(7)
S = collections.Counter(); T = collections.defaultdict(collections.Counter); ex = []
for b in random.sample(single, 20000):
    true = b["person"]
    saved = key2P[b["key"]]; key2P[b["key"]] = saved - {true}          # the person is not available as a candidate
    t, why = strict(b, exclude_person=true)
    n = len(key2P[b["key"]]); key2P[b["key"]] = saved
    tier = "0 (no other)" if n == 0 else "1" if n == 1 else "2-4" if n < 5 else "5+"
    res = "linked to another person" if t else why
    S[res] += 1; T[tier][res] += 1
    if t and len(ex) < 12:
        same_aff = None
        ex.append((b["pmcid"], f"{b['last']}, {b['fore']}", (b["afftext"] or "")[:70], P_name[t], t[1].rsplit("/",1)[-1]))
print("hidden single-article identified people:", sum(S.values()), dict(S))
print(f"false-merge rate (upper bound): {S['linked to another person']/sum(S.values()):.4f}")
for k in sorted(T): t = T[k]; print(f"  other same-key people {k}: hidden {sum(t.values())}, linked to another {t['linked to another person']} ({t['linked to another person']/sum(t.values()):.3%})")
for e in ex: print("  ", e)
