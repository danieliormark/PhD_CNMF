"""Score a reader's answers (JSONL, one object per question) against gold labels (RL-161).
    python3 gap_f2_score.py GOLD.json ANSWERS.jsonl
Reports: parse errors, coverage, agreement on the questions both answered decisively, the confusion table, and every
disagreement (so the owner or the assistant can look at them)."""
import json, sys, collections
gold = json.load(open(sys.argv[1]))["labels"]
ans, bad = {}, 0
for l in open(sys.argv[2]):
    l = l.strip()
    if not l:
        continue
    try:
        r = json.loads(l); ans[r["qid"]] = r
    except Exception:
        bad += 1
conf = collections.Counter((gold[q], ans[q]["verdict"] if q in ans else "missing") for q in gold)
dec = [q for q in gold if gold[q] in ("yes", "no") and q in ans and ans[q]["verdict"] in ("yes", "no")]
agree = sum(gold[q] == ans[q]["verdict"] for q in dec)
print(f"parse errors {bad}; answered {len(set(ans) & set(gold))}/{len(gold)}; decisive agreement {agree}/{len(dec)}")
for (g, a), n in sorted(conf.items()):
    print(f"  gold {g:9s} reader {a:9s} {n}")
for q in gold:
    a = ans.get(q, {}).get("verdict", "missing")
    if a != gold[q]:
        print("  DIFF", q, "gold", gold[q], "reader", a, ans.get(q, {}).get("reason", ""), ans.get(q, {}).get("note", ""))
