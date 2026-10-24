# Scratch, read-only: hide-and-recover precision of the strict B rule (owner-approved 2026-10-24).
exec(open("links_v2.py").read().split("U = [b for b in B if not b[\"person\"]]")[0])
import random
oa_by = collections.defaultdict(list)
for b in B:
    if b["oa_id"]: oa_by[b["oa_id"]].append(b)
SUSPECT = {("A", a) for a, bs in oa_by.items() if len({x["pm_orcid"] for x in bs if x["pm_orcid"]}) > 1}
SUSPECT_P = {find(k) for k in SUSPECT if k in parent}
def strict(b, exclude_person=None):
    """the strict B decision for byline b: (target person or None, reason)"""
    if not b["key"]: return None, "no key"
    cands = key2P.get(b["key"], set()); n = len(cands)
    ok = {}
    for c in cands:
        co, inst = set(), set()
        for art, (ins, cos) in P_ev[c].items():
            if art == b["pmcid"]: continue
            inst |= b["inst"] & ins
            co |= ((b["coauth"] - {c, exclude_person}) & cos) - SUSPECT_P
        if n >= 5: good = len(co) >= 2 and bool(inst)
        else: good = len(co) >= 2 or (len(co) >= 1 and bool(inst))
        if good: ok[c] = (co, inst)
    if len(ok) == 1: return next(iter(ok)), "linked"
    return None, "ambiguous" if ok else "no corroborated candidate"
random.seed(2026)
# people on 2+ articles (otherwise there is nothing to recover from)
arts_of = collections.defaultdict(set)
for b in B:
    if b["person"]: arts_of[b["person"]].add(b["pmcid"])
pool = [b for b in B if b["person"] and len(arts_of[b["person"]]) >= 2 and b["person"] not in SUSPECT_P]
S = collections.Counter(); T = collections.defaultdict(collections.Counter); wrong = []
for b in random.sample(pool, 20000):
    true = b["person"]
    # hide: this byline has no ID; its own article no longer counts as evidence for its person (evidence() skips it)
    t, why = strict(b, exclude_person=true)
    n = len(key2P.get(b["key"], ())) if b["key"] else 0
    tier = "1" if n <= 1 else "2-4" if n < 5 else "5+"
    res = "correct" if t == true else ("WRONG" if t else why)
    S[res] += 1; T[tier][res] += 1
    if t and t != true and len(wrong) < 10: wrong.append((b["pmcid"], f"{b['last']}, {b['fore']}", P_name.get(true), P_name.get(t), b["afftext"][:80]))
print("hidden bylines (people on 2+ articles):", sum(S.values()), dict(S))
linked = S["correct"] + S["WRONG"]
print(f"precision of links: {S['correct']/linked:.4f}   recall: {S['correct']/sum(S.values()):.3f}")
for k in ("1", "2-4", "5+"):
    t = T[k]; l = t["correct"] + t["WRONG"]
    print(f"  name key held by {k}: hidden {sum(t.values())}, linked {l}, wrong {t['WRONG']}, precision {t['correct']/l if l else float('nan'):.4f}")
for w in wrong: print("  WRONG", w)
