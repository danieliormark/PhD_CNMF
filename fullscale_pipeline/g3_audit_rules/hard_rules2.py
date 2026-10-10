"""Hard cases for the 2026-10-09 audit rules (hyphen properties, cousin grouping): parse each sentence with graphbrain as
G2 does and run the rule copy under several settings. Hyphen cases (H): `need` strings must appear in the output, `avoid`
strings must not. Grouping cases (G): every cousin that holds one of the `together` words must be one and the same cousin
holding all of them, whichever parent it belongs to.
    python hard_rules2.py hard_rules2.json SCRIPT
"""
import hashlib, importlib.util, json, os, sys
cases, script = json.load(open(sys.argv[1])), sys.argv[2]
SETTINGS = {"before": dict(XBASED="0", GROUP="0"), "hyphen+group1": dict(XBASED="hyphen", GROUP="1"),
            "hyphen+group2": dict(XBASED="hyphen", GROUP="2")}
def load(name, env):
    os.environ.update(dict(Q4="1", ONLY_MODE="vg2", HYPH_JOINED="1", **env))
    spec = importlib.util.spec_from_file_location(name, script); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mods = {k: load("g3_" + k.replace("+", "_"), v) for k, v in SETTINGS.items()}
from graphbrain.parsers import create_parser
parser = create_parser(lang="en", lemmas=True)
score = {k: 0 for k in mods}
for c in cases:
    parser.atom2token = {}
    p = parser.parse(c["text"])["parses"][0]
    words = sorted(([str(a), w, i] for a, (w, i) in p.get("atom2word", {}).items()), key=lambda x: x[2])
    uid = "PMCX.r1.L1.S1.U1"
    u = dict(uid=uid, hash=hashlib.sha1(f"{uid}|{c['text']}".encode()).hexdigest()[:12], text=c["text"],
             main_edge=str(p["main_edge"]), extra_edges=[str(x) for x in p.get("extra_edges", [])], atom2word=words)
    print(f"\n{c['id']}: {c['text']}\n   G2: {u['main_edge'][:240]}")
    for k, m in mods.items():
        rec, _ = m.curate_unit("PMCX", dict(sid="PMCX.r1.L1.S1", hash_final="0" * 12), u, m.focal_terms.Matcher(), True, "keep")
        par = " ".join(x["edge"] for x in rec["parents"]); cou = [x["edge"] for x in rec["cousins"]]
        if "together" in c:
            hold = {e for e in cou if any(f" {w}/" in e for w in c["together"])}
            ok = len(hold) == 1 and all(f" {w}/" in next(iter(hold)) for w in c["together"])
        else:
            allout = par + " " + " ".join(cou)
            ok = all(s in allout for s in c["need"]) and not any(s in allout for s in c["avoid"])
        score[k] += ok
        print(f"   [{k}] {'PASS' if ok else 'FAIL'}  {par[:250]}  ||  {' '.join(cou)[:250]}")
print("\n" + "  ".join(f"{k}: {v}/{len(cases)}" for k, v in score.items()))
