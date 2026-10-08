"""Q6: per article, P2 anchor (PLAIN) hits by term, and accepted generic/guarded hits; to find articles resting on one possibly homonymous term."""
import json, glob, collections, sys
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms
M = focal_terms.Matcher()
SH = sorted(glob.glob("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"))
out = {}
for f in SH:
    for l in open(f):
        a = json.loads(l)
        lines = [p.get("orig") or p.get("parsed") or "" for s in a["sentences"] for p in s.get("pieces", [])]
        full = "\n".join(lines)
        anch = collections.Counter(m.group(0).lower() for m in M.anchor_hits(full))
        anchored = bool(anch)
        acc = collections.Counter(); ctx = {}
        for ln in lines:
            for t, sp, txt in M.hits(ln, anchored)[0]:
                acc[t.lower()] += 1
                ctx.setdefault(t.lower(), ln[max(0, sp[0]-80):sp[1]+60])
        out[a["pmcid"]] = dict(anch=anch, acc=acc, ctx=ctx, nsent=len(lines))
json.dump(out, open("anchor_scan.json", "w"))
print("articles:", len(out), "with any accepted focal hit:", sum(1 for v in out.values() if v["acc"]))
