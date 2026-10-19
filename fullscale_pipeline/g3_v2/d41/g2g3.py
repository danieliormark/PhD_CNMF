import importlib.util, sys, json
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
g2 = load("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/graphbrain_parse_v3.py", "g2")
g3 = load("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/g3_v2/g3_curation_v2.py", "g3")
from graphbrain import hedge
from graphbrain.parsers import create_parser
parser = create_parser(lang="en", lemmas=True)
for i, t in enumerate(json.load(open(sys.argv[1]))):
    s = dict(sid=f"PMCX.r1.L{i+1}.S1", hash_final="0" * 12, text_final=t)
    rec, notes, seg = g2.process_sentence(parser, hedge, s, set(), {})
    print("\n#", t, "| G2 status", rec["status"], "cuts", seg, notes)
    for u in rec["units"]:
        print("  unit", u["uid"], "|", u["text"]); print("   edge", u["main_edge"])
        r3, _ = g3.curate_unit("PMCX", rec, u, g3.focal_terms.Matcher(), True, "keep")
        for p in r3["parents"]: print("   G3", p["edge"])
        for c in r3["cousins"]: print("   G3 cousin", c["edge"])
        if not r3["parents"]: print("   G3: no parent")
