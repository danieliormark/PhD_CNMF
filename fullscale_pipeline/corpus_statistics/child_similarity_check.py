# 2026-10-27 (owner): blind reading check of the child×child similarity tie selection before it is fixed. Psychology trial,
# child-level hyperedges (children and cousins merged by word set, G3 content key; demoted run for the 120 articles of the
# language-model demotion). Words = typed atoms incl. model names (owner); idf = ln(N / number of hyperedges with the word).
#   J  idf-weighted Jaccard  Σ idf(shared) / Σ idf(union)                (decided default)
#   C  idf-weighted cosine   Σ idf²(shared) / √(Σ idf²(A) · Σ idf²(B))    (comparison arm)
# Tie selection (decided): the 5 most similar hyperedges with similarity ≥ τ = 0.5, all candidates tied at the 5th value
# kept. Shown here: each sampled hyperedge's own selection only (ties it receives from other hyperedges' selections are
# not computed). Outputs in child_similarity_check/:
#   part_a.txt  50 random child hyperedges with 2+ words; for each, two lists in random order labelled X and Y (one per
#               measure), each tied hyperedge with its similarity
#   part_b.txt  60 random pairs with idf-weighted Jaccard between 0.4 and 0.5 (the τ question)
#   key.json    which of X / Y is which measure, per hyperedge (read only after judging)
import collections, glob, json, math, random

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
CS = FP + "corpus_statistics/"; OUT = CS + "child_similarity_check/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
import os; os.makedirs(OUT, exist_ok=True)
trial = {l.strip() for l in open(CS + "psychology_trial_pmcids.txt") if l.strip()}
recs, dem = [], set()
for f in glob.glob(PG + "g3_v2_lm_demotion/demoted/g3_test_*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r["pmcid"] in trial:
            dem.add(r["pmcid"]); recs.append(r)
for l in open(PG + "g3_v2/g3_test_v2.jsonl", encoding="utf-8"):
    p = l[11:l.index('"', 11)]
    if p in trial and p not in dem:
        recs.append(json.loads(l))
child = {}
for r in recs:
    for c in [c for p in r["parents"] for c in p["children"]] + r["cousins"]:
        child[c["content_key"]] = frozenset(a["atom"] for a in c["atoms"])
keys = sorted(child); sets = [child[k] for k in keys]; N = len(sets)
df = collections.Counter(w for s in sets for w in s)
idf = {w: math.log(N / df[w]) for w in df}
inv = collections.defaultdict(list)
for i, s in enumerate(sets):
    for w in s:
        inv[w].append(i)
sw = [sum(idf[w] for w in s) for s in sets]; sw2 = [sum(idf[w] ** 2 for w in s) for s in sets]


def sims(i):
    sh1, sh2 = collections.Counter(), collections.Counter()
    for w in sets[i]:
        for j in inv[w]:
            if j != i:
                sh1[j] += idf[w]; sh2[j] += idf[w] ** 2
    J = {j: v / (sw[i] + sw[j] - v) for j, v in sh1.items()}
    C = {j: sh2[j] / math.sqrt(sw2[i] * sw2[j]) for j in sh2}
    return J, C


def select(s, k=5, tau=0.5):
    c = sorted(((v, j) for j, v in s.items() if v >= tau), reverse=True)
    if len(c) <= k:
        return c
    kth = c[k - 1][0]
    return [x for x in c if x[0] >= kth - 1e-12]


show = lambda s: " ".join(sorted(a.rsplit("/", 1)[0] if a.endswith("/focal") else a.split("/")[0] + "/" + a.split("/")[1][0] for a in s))
rng = random.Random(27)
cand = [i for i, s in enumerate(sets) if len(s) >= 2]
key = {}
with open(OUT + "part_a.txt", "w", encoding="utf-8") as f:
    for n, i in enumerate(rng.sample(cand, 50), 1):
        J, C = sims(i); lists = [("J", select(J)), ("C", select(C))]; rng.shuffle(lists)
        key[n] = {"X": lists[0][0], "Y": lists[1][0]}
        f.write(f"\n[{n}] {show(sets[i])}\n")
        for lab, (_, sel) in zip("XY", lists):
            f.write(f"   {lab}: " + ("(no tie)" if not sel else "") + "\n")
            for v, j in sel:
                f.write(f"      {v:.2f}  {show(sets[j])}\n")
json.dump(key, open(OUT + "key.json", "w"), indent=0)
band = []
for i in rng.sample(cand, 3000):
    J, _ = sims(i)
    b = [j for j, v in J.items() if 0.4 <= v < 0.5]
    if b:
        band.append((i, rng.choice(b), J))
    if len(band) >= 60:
        break
with open(OUT + "part_b.txt", "w", encoding="utf-8") as f:
    for n, (i, j, J) in enumerate(band, 1):
        f.write(f"[{n}] {J[j]:.2f}   {show(sets[i])}   ||   {show(sets[j])}\n")
print("hyperedges", N, "| part A 50 | part B", len(band))
