import json, glob, re, collections, sys, csv
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms
M = focal_terms.Matcher()
EXCL = {r["pmcid"] for r in csv.DictReader(open("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/g3_scope_exclusions/scope_exclusions.csv"))}
RX = re.compile(r"(?i)(?<![A-Za-z0-9])large[-‐‑–][ ]?language(?:[-‐‑– ]models?|[-‐‑–]model(?:s|[-‐‑–]\w+)?)(?![A-Za-z0-9])")
forms = collections.Counter(); arts = set(); units = 0; unmatched = 0; newanchor = set(); ex = []
for f in sorted(glob.glob("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl")):
    for l in open(f):
        if "arge-" not in l and "arge‐" not in l and "arge‑" not in l: continue
        a = json.loads(l)
        if a["pmcid"] in EXCL: continue
        txt = [u["text"] for s in a["sentences"] for u in s.get("units", [])]
        hit = False
        for t in txt:
            ms = list(RX.finditer(t))
            if not ms: continue
            units += 1; hit = True
            acc = [sp for _, sp, _ in M.hits(t, True)[0]]
            for m in ms:
                forms[m.group(0)] += 1
                if not any(a0 <= m.start() < b0 for a0, b0 in acc):
                    unmatched += 1
                    if len(ex) < 6: ex.append(t[max(0, m.start()-60):m.end()+40])
        if hit:
            arts.add(a["pmcid"])
            if not any(M.anchor_hits(t) for t in txt): newanchor.add(a["pmcid"])
print("articles (kept corpus):", len(arts), "units:", units, "occurrences:", sum(forms.values()), "not matched by P2 now:", unmatched)
print("articles that would gain article evidence:", len(newanchor))
print(forms.most_common(15)); [print("  ", e) for e in ex]
