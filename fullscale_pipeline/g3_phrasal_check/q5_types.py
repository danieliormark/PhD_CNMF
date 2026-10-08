"""Q5: parser type of every particle in G2 units (shards 0,20,30,40, first 700 articles) and its fate in G3 (orig)."""
import json, re, collections, sys
from graphbrain import hedge
P = {'out', 'up', 'down', 'in', 'on', 'off', 'over'}
SH = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_%03d.jsonl"
PRE = sys.argv[1] if len(sys.argv) > 1 else "orig"
res = []
for s in [0, 20, 30, 40]:
    g3 = {}
    for l in open(f"{PRE}_{s}/g3_test_{s:03d}.jsonl"):
        r = json.loads(l)
        fused_pos, left_pos = set(), set()
        for p in r["parents"]:
            for k in p["children"]:
                for a in k["atoms"]:
                    root = a["atom"].split("/")[0]
                    for src, w, pos in a["src"]:
                        if w.lower() in P:
                            (left_pos if root in P else fused_pos).add(pos)
        for c in r["cousins"]:
            for a in c.get("atoms", []):
                root = a["atom"].split("/")[0]
                for src, w, pos in a["src"]:
                    if w.lower() in P:
                        (left_pos if root in P else fused_pos).add(pos)
        g3[r["uid"]] = (fused_pos, left_pos, r)
    n = 0
    for l in open(SH % s):
        art = json.loads(l); n += 1
        if n > 700: break
        for sen in art["sentences"]:
            for u in sen.get("units", []):
                if u["uid"] not in g3: continue
                fp, lp, r = g3[u["uid"]]
                edge = hedge(u["main_edge"])
                # parent edge of each particle atom: the smallest subedge containing it as a direct child
                par = {}
                def walk(e):
                    if e.is_atom(): return
                    for c in e:
                        if c.is_atom(): par.setdefault(str(c), str(e))
                        walk(c)
                walk(edge)
                for a, w, pos in u["atom2word"]:
                    if w.lower() not in P: continue
                    t = a.split("/")[1] if "/" in a else "?"
                    fate = "fused" if pos in fp else "left" if pos in lp else "dropped_or_absent"
                    res.append(dict(uid=u["uid"], word=w.lower(), type=t, pos=pos, fate=fate, parent=par.get(a, ""), text=u["text"]))
json.dump(res, open(f"q5_types_{PRE}.json", "w"))
c = collections.Counter((x["type"].split(".")[0], x["fate"]) for x in res)
types = sorted({k[0] for k in c}, key=lambda t: -sum(v for k, v in c.items() if k[0] == t))
print(f"{'type':8} {'fused':>7} {'left':>7} {'dropped/absent':>15}")
for t in types[:12]:
    print(f"{t:8} {c[(t,'fused')]:7} {c[(t,'left')]:7} {c[(t,'dropped_or_absent')]:15}")
