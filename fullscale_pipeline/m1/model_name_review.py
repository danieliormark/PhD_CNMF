# Review table of the draft model-name map (model_name_map.py) over the trial's focal atoms -> model_name_review.csv
import collections, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_name_map import canonical, ADAPTED, CANON
L1 = "/mnt/hum01-home01/p91688di/PhD_CNMF/plans/ticket107_layer1/"
__file__ = L1 + "model_levels.py"
exec(open(L1 + "model_levels.py").read().split("units = []")[0])
__file__ = os.path.abspath(sys.argv[0])
cnt = collections.defaultdict(set)
for r in recs:
    for c in [c for p in r["parents"] for c in p["children"]] + r["cousins"]:
        for a in c["atoms"]:
            if a["atom"].endswith("/C/focal"): cnt[a["atom"].split("/")[0]].add(r["pmcid"])
rows = []
for root, arts in cnt.items():
    out = {canonical(root, a) for a in arts}
    can = canonical(root)
    fam = level(can, "family") if can else None
    base = ADAPTED.get(fam, ("", ""))
    if can != root or base[0] or len(out) > 1:
        kind = ("not a model" if can is None else "identity (spelling/citation/adjective)" if can != root else "")
        if len(out) > 1: kind = "not a model in some articles"
        rows.append([root, len(arts), can or "", fam or "", kind, base[0], base[1], 0.5 if base[0] else ""])
rows.sort(key=lambda r: (r[4] == "", r[5], -r[1]))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_name_review.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["atom_root", "articles", "canonical_root", "family", "change", "adapted_from", "adapted_kind", "similarity_to_base"]); w.writerows(rows)
fams = collections.Counter(level(canonical(k) or k, "family") for k in cnt)
print("roots", len(cnt), "rows in review", len(rows), "| identity changes", sum(1 for r in rows if r[4].startswith("identity")),
      "| adapted", sum(1 for r in rows if r[5]), "| families before", len({level(k, 'family') for k in cnt}), "after", len(fams))
print("adapted by kind", collections.Counter(r[6] for r in rows if r[6]))
# leftovers: roots still containing a digit that are not a known version form
import re
left = sorted(k for k in cnt if canonical(k) and re.search(r"[a-z]\d", canonical(k)) and canonical(k) not in ("kogpt2", "aragpt2"))
print("digit-glued roots left:", left)
unl = sorted((len(cnt[k]), k) for k in cnt if canonical(k) and level(canonical(k), "family") not in ADAPTED
             and level(canonical(k), "family") not in {"bert","roberta","gpt","llama","gemini","gemma","claude","qwen","deepseek","mistral","palm","grok","llm","language_model","transformer_model"})
print("families neither base nor adapted (kept as their own family):", [k for _, k in unl[::-1]])
