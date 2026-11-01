# Sentence context for ambiguous focal model atoms (hand list of model-name mappings).
import os, re, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w_stats.py")).read().split('print("articles"')[0])
want = set(sys.argv[1:]); seen = {w: 0 for w in want}
for r in recs:
    atoms = {a["atom"].split("/")[0] for p in r["parents"] for c in p["children"] for a in c["atoms"] if a["atom"].endswith("/C/focal")}
    atoms |= {a["atom"].split("/")[0] for c in r["cousins"] for a in c["atoms"] if a["atom"].endswith("/C/focal")}
    for w in atoms & want:
        if seen[w] < 2: seen[w] += 1; print(f"[{w}] {r['text'][:230]}")
