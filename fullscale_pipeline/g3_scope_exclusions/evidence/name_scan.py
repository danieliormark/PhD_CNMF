"""Q6: every occurrence of ProGen, BioBridge, PaLM 2, ESM-2 (P2 PLAIN, case-insensitive) with exact case and context."""
import json, glob, re, collections
T = {"progen": r"ProGen", "biobridge": r"BioBridge", "palm 2": r"PaLM 2", "esm-2": r"ESM-2"}
RX = {k: re.compile(r'(?i)(?<![a-zA-Z0-9])' + re.escape(v) + r'(?![a-zA-Z0-9])') for k, v in T.items()}
out = collections.defaultdict(list)
for f in sorted(glob.glob("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl")):
    for l in open(f):
        low = l.lower()
        if not any(k in low for k in ("progen", "biobridge", "palm 2", "esm-2")): continue
        a = json.loads(l)
        full = " ".join(p.get("orig") or "" for s in a["sentences"] for p in s.get("pieces", []))
        for k, rx in RX.items():
            for m in rx.finditer(full):
                out[k].append((a["pmcid"], m.group(0), full[max(0, m.start()-70):m.end()+60]))
json.dump(out, open("name_scan.json", "w"), indent=0)
for k, L in out.items():
    print(k, "occurrences", len(L), "articles", len({x[0] for x in L}), "surface:", collections.Counter(x[1] for x in L).most_common())
