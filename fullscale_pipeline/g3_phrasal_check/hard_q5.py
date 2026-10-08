"""Parse hard_q5.json with graphbrain as G2 does; run the original G3 script and the Q5 flip copy; print both."""
import hashlib, importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
V = {"orig": "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_curation_test.py", "flip": f"{HERE}/../diagnostics/g3_stoplist/q5/g3_q5flip.py", "fix": f"{HERE}/../diagnostics/g3_stoplist/q5/g3_q5fix.py"}
def load(n, p):
    spec = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
cases = json.load(open(sys.argv[1]))
from graphbrain.parsers import create_parser
parser = create_parser(lang="en", lemmas=True)
mods = {k: load("g3_" + k, p) for k, p in V.items()}
out = []
for c in cases:
    parser.atom2token = {}
    p = parser.parse(c["text"])["parses"][0]
    words = sorted(([str(a), w, i] for a, (w, i) in p.get("atom2word", {}).items()), key=lambda x: x[2])
    uid = "PMCX.r1.L1.S1.U1"
    u = dict(uid=uid, hash=hashlib.sha1(f"{uid}|{c['text']}".encode()).hexdigest()[:12], text=c["text"],
             main_edge=str(p["main_edge"]), extra_edges=[str(x) for x in p.get("extra_edges", [])], atom2word=words)
    row = {"id": c["id"], "text": c["text"], "edge": u["main_edge"]}
    print(f"{c['id']}: {c['text']}\n   G2: {u['main_edge'][:230]}")
    for k, m in mods.items():
        rec, _ = m.curate_unit("PMCX", dict(sid="PMCX.r1.L1.S1", hash_final="0" * 12), u, m.focal_terms.Matcher(), True, "keep")
        txt = " ".join(x["edge"] for x in rec["parents"]) + " || " + " ".join(x["edge"] for x in rec["cousins"])
        row[k] = txt
        print(f"   [{k}] {txt[:330]}")
    out.append(row)
json.dump(out, open(sys.argv[2], "w"), indent=1)
