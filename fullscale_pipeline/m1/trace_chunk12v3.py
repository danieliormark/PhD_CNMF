# Prerequisite check of chunk12v3 on real trial records (ticket 107): a subset of articles (the first N of the sorted
# trial list plus named articles), built with the builder's functions, then
#   (1) identity: child and parent hyperedges whose atoms are unchanged by the rules keep G3's content key;
#   (2) the false-positive Grover units of the trial are removed, the RoBERT mapping is applied;
#   (3) cells recomputed from the raw records for random child hyperedges and parents;
#   (4) W by brute force and by prefix filtering agree; timing, to estimate the full build.
# Nothing is written. Run: python trace_chunk12v3.py [N]
import collections, math, random, sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk12v3 as B
import model_name_map as mnm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
EXTRA = ["PMC12961534", "PMC13318610", "PMC12370911", "PMC10308283"]
trial = sorted(l.strip() for l in open(B.INPUTS["trial_list"]) if l.strip())
sub = set(trial[:N]) | set(EXTRA)
t0 = time.time(); recs = B.load_records(sub); print(f"{len(sub)} articles, {len(recs)} units, loaded in {time.time() - t0:.0f}s")
log = {}; units = B.semantic_units(recs, log); sem = B.build_semantic(units)
print("units dropped:", log["units_dropped"])

# (1) identity
same_c = diff_c = 0; same_p = diff_p = 0; changed_c = 0
for r in recs:
    for p in r["parents"]:
        kids = [B.clean_atoms(c["atoms"], r["pmcid"]) for c in p["children"]]
        unchanged = all(frozenset(a["atom"] for a in c["atoms"]) == k for c, k in zip(p["children"], kids))
        if unchanged:
            new = B.content_key(B.content_key(s) for s in kids)
            same_p += new == p["content_key"]; diff_p += new != p["content_key"]
        for c, k in zip(p["children"], kids):
            if frozenset(a["atom"] for a in c["atoms"]) == k:
                same_c += B.content_key(k) == c["content_key"]; diff_c += B.content_key(k) != c["content_key"]
            else: changed_c += 1
print(f"(1) unchanged child hyperedges with G3's key {same_c}, different {diff_c}; changed by the rules {changed_c}; "
      f"unchanged parents with G3's key {same_p}, different {diff_p}")
assert diff_c == 0 and diff_p == 0

# (2) Grover, RoBERT
fp = [u for u, _ in log["units_dropped"]]
print("(2) dropped units:", fp)
for pm in EXTRA:
    roots = sorted({a.split("/")[0] for k in sem["children"] for a in sem["child_sets"][k] if a.endswith("/C/focal")
                    and any(u["pmcid"] == pm for u in units)})
cs_by_art = collections.defaultdict(set)
for u in units:
    for _, kids in u["parents"]:
        for s in kids: cs_by_art[u["pmcid"]] |= {a for a in s if a.endswith("/C/focal")}
for pm in EXTRA: print(f"    {pm} focal atoms: {sorted(cs_by_art[pm])}")
assert not any("grover" in a for pm in EXTRA[:2] for a in cs_by_art[pm])
assert all(why == "no focal atom left" for _, why in log["units_dropped"]), log["units_dropped"]
pl = [u for u in units if not u["parents"]]
print(f"    units kept for their cousins only (no parent of their own): {len(pl)}, cousin ties to parents of the sentence "
      f"{sum(len(p) for u in pl for _, p in u['cousins'])}")
assert "roberta/C/focal" in cs_by_art["PMC12370911"] and "robert/C/focal" not in cs_by_art["PMC12370911"]

# (3) cells recomputed from the raw records (independent code path)
random.seed(3)
arts = sem["articles"]; ai = {a: i for i, a in enumerate(arts)}; ci = {k: i for i, k in enumerate(sem["children"])}
tf = collections.Counter(); df = collections.defaultdict(set)
for r in recs:
    if r["uid"] in fp: continue   # every dropped unit, whatever the reason
    for p in r["parents"]:
        for c in p["children"]:
            s = B.clean_atoms(c["atoms"], r["pmcid"])
            if s: tf[(B.content_key(s), r["pmcid"])] += 1; df[B.content_key(s)].add(r["pmcid"])
    for c in r["cousins"]:
        s = B.clean_atoms(c["atoms"], r["pmcid"])
        if s: tf[(B.content_key(s), r["pmcid"])] += 1; df[B.content_key(s)].add(r["pmcid"])
CA = sem["CA"].tocsr(); worst = 0
for a in random.sample(arts, 10):
    raw = {k: math.log1p(t) * (math.log(len(arts) / (len(df[k]) + 1)) + 1) for (k, aa), t in tf.items() if aa == a}
    n = math.sqrt(sum(v * v for v in raw.values()))
    for k, v in raw.items(): worst = max(worst, abs(CA[ai[a], ci[k]] - v / math.sqrt(n)))
    assert CA[ai[a]].nnz == len(raw)
print(f"(3) child×article: 10 articles recomputed, largest difference {worst:.2e}")
ex = random.choice(arts); k = max((k for (k, aa) in tf if aa == ex), key=lambda k: tf[(k, ex)])
print(f"    example {ex}: child hyperedge {sorted(sem['child_sets'][k])}, tf {tf[(k, ex)]}, articles {len(df[k])}, "
      f"idf_article {sem['idf_article'][ci[k]]:.3f}, cell {CA[ai[ex], ci[k]]:.4f}")
pk_n, pk_c = collections.Counter(), collections.Counter()
for u in units:
    for pk, kids in u["parents"]:
        for s in kids: pk_n[(B.content_key(s), pk)] += 1
    for s, pks in u["cousins"]:
        for pk in pks: pk_c[(B.content_key(s), pk)] += 1
pairs = set(pk_n) | set(pk_c); P = len({p for _, p in pairs}); dfp = collections.Counter(k for k, _ in pairs)
PC = sem["PC"].tocsr(); pi = {p: i for i, p in enumerate(sem["parents"])}; worst = 0
for p in random.sample(sem["parents"], 20):
    raw = {k: (math.log(P / (dfp[k] + 1)) + 1) * math.log1p(pk_n[(k, pp)] + B.COUSIN * pk_c[(k, pp)]) for (k, pp) in pairs if pp == p}
    n = math.sqrt(sum(v * v for v in raw.values()))
    for k, v in raw.items(): worst = max(worst, abs(PC[pi[p], ci[k]] - v / math.sqrt(n)))
print(f"(3) child×parent: 20 parent hyperedges recomputed, largest difference {worst:.2e}")

# (4) W
sets = [sem["child_sets"][k] for k in sem["children"]]
ws = B.WordSpace(sets)
t1 = time.time(); Wb, sb = B.w_brute_force(ws); tb = time.time() - t1
t1 = time.time(); Wp, spf = B.w_prefix_filter(ws); tp = time.time() - t1
same = Wb.keys() == Wp.keys() and all(Wb[k] == Wp[k] for k in Wb)
print(f"(4) W on {len(sets)} child hyperedges: brute force {len(Wb)} ties, {sb:,} pairs, {tb:.0f}s; "
      f"prefix filter {len(Wp)} ties, {spf:,} pairs, {tp:.0f}s; identical {same}")
soft = [(i, j) for (i, j) in Wb if abs(Wb[(i, j)] - (lambda o: o / (ws.weight[i] + ws.weight[j] - o))(sum(ws.idf[a] for a in ws.words[i] & ws.words[j]))) > 1e-12]
print(f"    ties that need the soft model-name match: {len(soft)}")
for i, j in soft[:4]: print("     ", sorted(ws.words[i]), "~", sorted(ws.words[j]), round(Wb[(i, j)], 3))
assert same
print("all checks passed")
