# Articles with all / at least one OpenAlex-identified author, per table row; and a profile
# of the bylines without an OpenAlex author ID.
import json, sys, collections, re
CODE = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics"
DATA = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors"
sys.path.insert(0, CODE)
from fullscale_analysis import classify_subject, PLAIN_CANON, EXTRA_CANON
from family_map import FAMILY, family_of_term
V = {k: w for k, (m, w) in json.load(open(f"{DATA}/openalex_verified.json")).items()}
R = json.load(open(f"{DATA}/fullscale_results.json"))
FAM_L = {k.lower(): v for k, v in FAMILY.items()}
fam = family_of_term
arts, fc = {}, collections.Counter()
for line in open("/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/focal_sentences_v2.jsonl"):
    d = json.loads(line); arts[d["pmcid"]] = [{fam(t) for t in s["terms"]} for s in d["sentences"]]
    for s in arts[d["pmcid"]]:
        for f in s: fc[f] += 1
Q = {f.lower() for f, _ in fc.most_common(20)} | {"llm (generic term)", "bert (bare/unspecified)"}
A = {p for p in arts if classify_subject(R[p]["subjects"]) in ("article_like", "review")}
B = {p for p, ss in arts.items() if any(any(f.lower() in Q for f in s) for s in ss)}
for label, ids in [("All", set(arts)), ("(a)", A), ("(b)", B), ("(a) and (b)", A & B)]:
    found = [p for p in ids if V.get(p)]
    allid = sum(1 for p in found if V[p]["authors"] and all(a["id"] for a in V[p]["authors"]))
    one = sum(1 for p in found if any(a["id"] for a in V[p]["authors"]))
    none = sum(1 for p in found if V[p]["authors"] and not any(a["id"] for a in V[p]["authors"]))
    print(f"{label:12s} articles {len(ids)} in OpenAlex {len(found)} all-identified {allid} >=1-identified {one} none-identified {none}")
noid = [(p, a["raw"]) for p, w in V.items() if w for a in w["authors"] if not a["id"]]
per = collections.Counter(p for p, _ in noid)
group = re.compile(r"(?i)group|consortium|collaborat|committee|network|investigators|team|society|initiative|working|study|association|council|panel|task force")
print("bylines without ID", len(noid), "in", len(per), "articles; exactly one missing in", sum(v == 1 for v in per.values()),
      "; group-like names", sum(1 for _, r in noid if r and group.search(r)))
