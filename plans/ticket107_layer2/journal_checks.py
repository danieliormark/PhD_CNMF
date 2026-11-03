# Ticket 107, L2.1 (2026-11-02): (1) same-subfield rate of within-journal article pairs (OpenAlex primary subfield) by
# journal size, against random pairs: sum_c c(c-1) / (n(n-1)) per journal, pooled by pairs; (2) owner's popularity
# check: per journal (2015-2025, working corpus v2), psychology-primary articles (popularity) and their share of the
# journal's working-corpus articles, for journals with >= 10 corpus articles. Read-only; journal IDs are NLM IDs.
import csv, collections, numpy as np
CS = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/"
trial = {l.strip() for l in open(CS + "psychology_trial_pmcids.txt") if l.strip()}
wc = {l.strip() for l in open(CS + "working_corpus_v2_pmcids.txt") if l.strip()}
rows = [r for r in csv.DictReader(open(CS + "subset_candidates_by_article.csv")) if r["pmcid"] in wc]
J = collections.defaultdict(list)
for r in rows:
    if r["pmcid"] in trial: J[r["nlm_id"]].append(r["subfield"])
def same(x):
    c = np.array(list(collections.Counter(x).values()), float); k = c.sum(); return (c * (c - 1)).sum() / (k * (k - 1))
print("(1) random pair of trial articles, same subfield:", round(same([s for v in J.values() for s in v]), 3))
for lo, hi in [(2, 3), (4, 9), (10, 30), (31, 9999)]:
    num = den = 0
    for v in J.values():
        k = len(v)
        if lo <= k <= hi: num += same(v) * k * (k - 1); den += k * (k - 1)
    print(f"    pairs within journals of {lo}-{hi} trial articles: {num / den:.3f}")
big = max(J, key=lambda j: len(J[j])); print(f"    largest journal ({len(J[big])} articles): {same(J[big]):.3f}")
rows = [r for r in rows if r["year"] and 2015 <= int(r["year"]) <= 2025 and r["nlm_id"]]
C = collections.Counter(r["nlm_id"] for r in rows); P = collections.Counter(r["nlm_id"] for r in rows if r["field"] == "Psychology")
print("(2) journals with >= 10 corpus articles, 2015-2025: popularity bin | journals | pooled psychology share | psychology-majority journals")
for lo, hi in [(1, 1), (2, 3), (4, 9), (10, 30), (31, 9999)]:
    sel = [j for j in P if lo <= P[j] <= hi and C[j] >= 10]
    print(f"    {lo}-{hi}: {len(sel)} | {sum(P[j] for j in sel) / max(1, sum(C[j] for j in sel)):.2f} | {sum(P[j] / C[j] >= .5 for j in sel)}")
