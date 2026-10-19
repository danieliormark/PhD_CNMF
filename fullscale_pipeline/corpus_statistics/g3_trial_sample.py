# Ticket 107 P1.D (owner, 2026-10-19): hand-read sample of G3 output on the trial slices. 30 units drawn at random
# (seed 107) from the 2020 and 2021 trial articles, each rendered with its sentence, parents (verb group, focal
# hyperedge, sibling arguments) and cousins, plus each atom's source words. Output: ../g3_trial_sample/sample_30.md.
import json, random
OD = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/author_coverage/"
T = json.load(open(OD + "trial_p1_checks.json"))["trial_pmcids"]
TR = set(T["2020"]) | set(T["2021"]) | {"PMC7815194", "PMC8051288", "PMC8523728", "PMC8666839", "PMC8752034"}
units = [json.loads(l) for l in open("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/g3_test_v1.jsonl")
         if l[11:22] in TR or json.loads(l)["pmcid"] in TR] if False else []
for l in open("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/g3_test_v1.jsonl"):
    r = json.loads(l)
    if r["pmcid"] in TR:
        units.append(r)
S = random.Random(107).sample(units, 30)
src = lambda c: " ".join(f"{a['atom'].split('/')[0]}←'{' '.join(s[1] for s in a['src'])}'" for a in c["atoms"])
L = ["# G3 hand-read sample, trial slices 2020/2021 (30 units, seed 107)", "",
     f"Drawn from {len(units):,} units of {len(TR)} trial articles. For each unit: the sentence, then each parent "
     "(one per focal occurrence) with its children, then the cousins (other clauses) with the parents they attach to. "
     "After each hyperedge: atom ← source words.", ""]
for n, r in enumerate(S, 1):
    L += [f"## {n}. {r['uid']}", "", f"> {r['text']}", ""]
    for p in r["parents"]:
        L.append(f"- **parent** {p['id'].rsplit('.', 1)[-1]}")
        for c in p["children"]:
            L.append(f"  - `{c['edge']}` — {src(c)}")
    for c in r["cousins"]:
        L.append(f"- cousin `{c['edge']}` → {', '.join(x.rsplit('.', 1)[-1] for x in c['parents'])} — {src(c)}")
    L += ["", "Reading: ", ""]
open("../g3_trial_sample/sample_30.md", "w").write("\n".join(L))
print(len(units), "units; wrote sample_30.md")
