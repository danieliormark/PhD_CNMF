# Full-scale summary table restricted to articles whose sentences contain >=1 focal item
# meeting the row's criterion (official P2 focal sentences, focal_sentences_v2.jsonl),
# with distinct-author counts from OpenAlex author IDs and from ORCID.
import json, sys, collections
CODE = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics"
DATA = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors"
sys.path.insert(0, CODE)
from fullscale_analysis import classify_subject, PLAIN_CANON, EXTRA_CANON
from family_map import FAMILY, family_of_term

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
G2 = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/g2_parsed_v3.jsonl"
R = json.load(open(f"{DATA}/fullscale_results.json"))
WC = json.load(open(f"{DATA}/wordcount_allregions.json"))
H = json.load(open(f"{DATA}/headers.json"))
OA = {k: (w if w else None) for k, (m, w) in json.load(open(f"{DATA}/openalex_verified.json")).items()}

fam = family_of_term

# per-article focal sentences: (sid, families, source)
arts = {}
fam_sent = collections.Counter()
for line in open(LCS + "focal_sentences_v2.jsonl"):
    d = json.loads(line)
    sents = []
    for s in d["sentences"]:
        fs = {fam(t) for t in s["terms"]}
        sents.append((s["sid"], fs, s.get("source")))
        for f in fs:
            fam_sent[f] += 1
    arts[d["pmcid"]] = sents

units = {}
for line in open(G2):
    d = json.loads(line)
    for s in d["sentences"]:
        units[s["sid"]] = len(s.get("units", []))

PREV_TOP20 = ["LLM (generic term)", "GPT (OpenAI)", "BERT (bare/unspecified)", "language model (generic term)",
              "Transformer (generic term)", "Gemini (Google)", "GPT (bare/unspecified)", "DeepSeek",
              "Claude (Anthropic)", "Llama (Meta)", "RoBERTa (generic/base)", "Bard (Google)", "BioBERT",
              "Mistral", "ESM-2", "Qwen", "Grok (xAI)", "DNABERT", "GPT-4V", "Gemma (Google)"]
top20_official = [f for f, _ in fam_sent.most_common(20)]
QUAL = {f.lower() for f in top20_official} | {"llm (generic term)", "bert (bare/unspecified)"}
print("top-20 families by focal-sentence count (official P2 terms):")
for f, n in fam_sent.most_common(22):
    print(f"   {n:>7}  {f}{'' if f.lower() in {x.lower() for x in PREV_TOP20} else '   <- not in the earlier top-20'}")
print("earlier top-20 families missing from official top-20:",
      [f for f in PREV_TOP20 if f.lower() not in {x.lower() for x in top20_official}])

art_type = {k: classify_subject(v["subjects"]) for k, v in R.items()}
missing_R = [k for k in arts if k not in R]
print("focal articles", len(arts), "missing from text-corpus record", len(missing_R))


def row(label, pmcids, qualifying):
    """qualifying(families) -> bool, applied per focal sentence."""
    keep, fsent, hyper, coref_only = [], 0, 0, 0
    for p in pmcids:
        q = [(sid, src) for sid, fs, src in arts[p] if qualifying(fs)]
        if not q:
            continue
        keep.append(p)
        fsent += len(q); hyper += sum(units.get(sid, 0) for sid, _ in q)
        coref_only += all(src == "coref" for _, src in q)
    words = sum(WC[p]["words"] for p in keep)
    sents = sum(WC[p]["sentences"] for p in keep)
    bylines = sum(H[p]["n_author_lines"] for p in keep if H.get(p))
    oa_found = [p for p in keep if OA.get(p)]
    oa_auth = {a["id"] for p in oa_found for a in OA[p]["authors"] if a["id"]}
    oa_bylines = sum(len(OA[p]["authors"]) for p in oa_found)
    oa_noid = sum(1 for p in oa_found for a in OA[p]["authors"] if not a["id"])
    orcids = {o for p in keep if H.get(p) for o in H[p]["orcids"]}
    return dict(row=label, articles=len(keep), coref_only=coref_only, words=words, sentences=sents,
                focal_sentences=fsent, hyperedges=hyper, bylines_header=bylines,
                articles_in_openalex=len(oa_found), bylines_openalex=oa_bylines, bylines_without_id=oa_noid,
                distinct_authors_openalex=len(oa_auth), distinct_orcids=len(orcids))


ALL = list(arts)
A = [p for p in ALL if art_type.get(p) in ("article_like", "review")]
anyq = lambda fs: bool(fs)
bq = lambda fs: any(f.lower() in QUAL for f in fs)
rows = [row("All", ALL, anyq), row("(a) article/review only", A, anyq),
        row("(b) LLM + BERT + top-20", ALL, bq), row("(a) and (b)", A, bq)]
json.dump(rows, open(f"{DATA}/table.json", "w"), indent=1)
for r in rows:
    print(r)

import numpy as np
diff = np.array([len(OA[p]["authors"]) - H[p]["n_author_lines"] for p in arts if OA.get(p) and H.get(p)])
print("per-article OpenAlex authors minus PMC header author lines: equal %.1f%%, OA fewer %.1f%%, OA more %.1f%%, median %d, p5 %d, p95 %d" % (
    100*np.mean(diff == 0), 100*np.mean(diff < 0), 100*np.mean(diff > 0), np.median(diff), np.percentile(diff, 5), np.percentile(diff, 95)))
big = sorted(((len(OA[p]["authors"]) - H[p]["n_author_lines"], p) for p in arts if OA.get(p) and H.get(p)))[:4]
print("largest shortfalls (OA - header, pmcid):", big)
