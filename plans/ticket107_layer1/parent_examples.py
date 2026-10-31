# Source sentences of the most frequent parent hyperedges (owner's question: is "(base llm)" from "LLM-based"?).
import collections, json, os, random, re
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w_stats.py")).read().split('print("articles"')[0])
by = collections.defaultdict(list)
for r in recs:
    for p in r["parents"]: by[p["content_key"]].append((r["text"], p["edge"]))
top = sorted(by, key=lambda k: -len(by[k]))[:6]
random.seed(1)
for k in top:
    print("\n==", len(by[k]), "units; edge:", by[k][0][1][:160])
    for t, _ in random.sample(by[k], 4): print("   -", t[:200])
