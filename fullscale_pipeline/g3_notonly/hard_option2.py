"""Option 2 hard cases: is not_only / but_also present in the curated output as expected? Modes: keep (0) and paired."""
import hashlib, importlib.util, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
def load(name, mode, path):
    os.environ.update({"Q4": "1", "ONLY_MODE": "vg2", "NOTONLY_DROP": mode})
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
from graphbrain.parsers import create_parser
parser = create_parser(lang="en", lemmas=True)
mods = {"keep": load("k", "0", f"{HERE}/../diagnostics/g3_notonly/g3_q4drop.py"), "paired": load("p", "paired", f"{HERE}/../diagnostics/g3_notonly/g3_q4drop.py"),
        "rl103": load("r", "paired", f"{HERE}/../diagnostics/g3_notonly/g3_q4drop_rl103.py")}
cases = json.load(open(sys.argv[1]))
score = {k: 0 for k in mods}; fused = 0
for c in cases:
    parser.atom2token = {}
    p = parser.parse(c["text"])["parses"][0]
    words = sorted(([str(a), w, i] for a, (w, i) in p.get("atom2word", {}).items()), key=lambda x: x[2])
    uid = "PMCX.r1.L1.S1.U1"
    u = dict(uid=uid, hash=hashlib.sha1(f"{uid}|{c['text']}".encode()).hexdigest()[:12], text=c["text"], main_edge=str(p["main_edge"]),
             extra_edges=[str(x) for x in p.get("extra_edges", [])], atom2word=words)
    line = f"{c['id']:4} want not_only={'kept' if c['not_only'] else 'gone'} but_also={'kept' if c['but_also'] else 'gone'} |"
    for k, m in mods.items():
        rec, _ = m.curate_unit("PMCX", dict(sid="PMCX.r1.L1.S1", hash_final="0" * 12), u, m.focal_terms.Matcher(), True, "keep")
        txt = " ".join(x["edge"] for x in rec["parents"] + rec["cousins"])
        if not txt.strip():
            line += f" {k}: EMPTY |"; continue
        no, ba = "not_only/M/en" in txt, "but_also/M/en" in txt
        if k == "keep":
            fused += 1
            line += f" keep: fused not_only={no} but_also={ba} |"
        else:
            ok = (no == c["not_only"]) and (ba == c["but_also"]); score[k] += ok
            line += f" {k}: {'OK' if ok else 'FAIL'} ({'n' if no else '-'}{'b' if ba else '-'}) |"
    print(line + (f"  [{c['note']}]" if c.get("note") else ""))
print("score:", {k: f"{v}/{len(cases)}" for k, v in score.items() if k != "keep"})
