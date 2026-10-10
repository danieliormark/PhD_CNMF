"""§6 item 1, before implementing: every auxiliary-typed "be" (type Mv, a form of be) in a G2 shard, classified by the
first content word after it (adverbs, negation, "been"/"being" skipped): a passive main verb (type P, argument roles with
'p'), another main verb (progressive etc.), or a non-verb (adjective or noun: the mistyped-copula signature, "cannot be
absorbable"). Also: is that next verb in the same connector as the "be"? RL-112.
    python measure_be.py G2_SHARD.jsonl"""
import json, random, sys, collections
sys.setrecursionlimit(20000)
from graphbrain import hedge
BE = {"be", "is", "are", "was", "were", "been", "being", "am", "'s", "'re", "’s"}
SKIP = ("M", "Mn", "Mv", "Mm", "Mi")
cls, ex = collections.Counter(), collections.defaultdict(list)
for line in open(sys.argv[1]):
    a = json.loads(line)
    for s in a["sentences"]:
        for u in s["units"]:
            if not u["main_edge"]: continue
            tok = {p: (at, w) for at, w, p in u["atom2word"]}
            for p in sorted(tok):
                at, w = tok[p]
                t = at.split("/")[1] if "/" in at else ""
                if not (t.startswith("Mv") and w.lower() in BE): continue
                q, nxt = p + 1, None
                while q <= max(tok):
                    if q in tok:
                        a2, w2 = tok[q]; t2 = a2.split("/")[1]
                        if t2.startswith("Mv") and w2.lower() in BE or t2.split(".")[0] in SKIP:
                            q += 1; continue
                        nxt = (a2, w2, t2); break
                    q += 1
                if nxt is None: k = "nothing after it"
                elif nxt[2].startswith("P"):
                    f = nxt[2].split("."); roles = f[1] if len(f) > 1 and f[1][:1] not in "<|-" else ""   # no role field in 'Pd.<pf--'
                    k = "passive main verb (kept)" if "p" in roles else "other main verb (progressive etc.; dropped)"
                else:
                    k = f"non-verb next ({nxt[2][:2]}): mistyped-copula signature"
                cls[k] += 1
                ex[k].append(f"{w} … {nxt[1] if nxt else ''}  ||  {u['text'][:140]}")
n = sum(cls.values())
for k, v in cls.most_common(): print(f"{v:6d} {v/n:6.1%}  {k}")
random.seed(3)
for k in cls:
    print("\n--", k)
    for x in random.sample(ex[k], min(8, len(ex[k]))): print("   ", x)
