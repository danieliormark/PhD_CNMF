"""Run the G3 test script (unchanged) on the 6 corrected units, before and after the correction; print focal mentions."""
import importlib.util, json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_scope_exclusions as B
spec = importlib.util.spec_from_file_location("g3", "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_curation_test.py")
g3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g3)
C = B.load_corrections(os.path.join(B.HERE, "spelling_corrections.csv"))
m = g3.focal_terms.Matcher(); ok = 0
for f in sorted(glob.glob(B.SHARDS)):
    for line in open(f):
        if not any(u.split(".")[0] in line[:40] for u in C): continue
        art = json.loads(line)
        for s in art["sentences"]:
            for u in s.get("units", []):
                if u["uid"] not in C: continue
                before = g3.focal_mentions(m, u["text"], u["atom2word"], True)
                v = B.apply_corrections(u, C)
                after = g3.focal_mentions(m, v["text"], v["atom2word"], True)
                rec, _ = g3.curate_unit(art["pmcid"], s, v, m, True, "keep")
                fb = sorted({x[2] for x in before}); fa = sorted({x[2] for x in after})
                good = "llm" in fa and any(x[1].lower().startswith("large language") for x in after)
                edges = " ".join([p["edge"] for p in rec["parents"]] + [c["edge"] for c in rec["cousins"]])
                stray = [w for w in ("languge", "langue", "langaue", "learning") if w + "/" in edges]
                good = good and not stray
                print("   stray atoms:", stray, "|", edges[:220])
                ok += good
                print(u["uid"], "| before:", [x[1] for x in before], "| after:", [x[1] for x in after], fa, "parents:", len(rec["parents"]), "PASS" if good else "FAIL")
print(ok, "of", sum(len(v) for v in C.values()), "pass")
