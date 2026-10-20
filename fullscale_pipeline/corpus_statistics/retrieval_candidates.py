# Candidate articles for the corpus extension (owner 2026-10-20, RL-155): PMC open access, publication 2015-01..2026-09,
# body text and title/abstract, minus R1's targets. Term sets:
#   OLD_PLAIN  R1's names and phrases, except the names that need article evidence (focal_terms.NEED_EVIDENCE: guarded
#              names such as Claude, Grok, RITA, GPN, GROVER — like GPT/Gemini they never select an article on their own)
#   KEPT       spellings of listed terms and new model names kept after the drops (kept_terms.json), not flagged
#   FLAGGED    terms with body-text hits before 2019 (release_check.tsv) plus BERT, RoBERTa, "language model(s)":
#              searched twice — plain, and with a context word in the body text (CONTEXT; for the "language model"
#              terms without the language-model words themselves)
# Version A = OLD_PLAIN | KEPT | FLAGGED plain; version B = OLD_PLAIN | KEPT | FLAGGED with context.
#   python3 retrieval_candidates.py GAP_AUDIT_DIR
import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = sys.argv[:2] + ["x", "x", "x", "Title/Abstract"]          # retrieval_gap_query reads FIELD from argv[4]
import retrieval_gap_query as Q
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms

GA = sys.argv[1].rstrip("/") + "/"
LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
CONTEXT = ["transformer", "transformers", "language model", "language models", "natural language processing", "NLP", "pretrained",
           "pre-trained", "fine-tuned", "fine-tuning", "deep learning", "neural network", "neural networks", "machine learning",
           "embedding", "embeddings", "attention mechanism", "tokenizer", "tokenization"]
LM_CONTEXT = [c for c in CONTEXT if "language model" not in c]

r1 = open("/mnt/hum01-rds/Basov/p91688di/query_pmc_entrez.py").read()
old = re.findall(r'"([^"]+)"\[(?:Title/Abstract|Text Word)\]', r1)
need_ev = {n.lower() for n in focal_terms.NEED_EVIDENCE} | {s.lower() for v in focal_terms.GUARDED_NAMES.values() for s in v.split("|")}
OLD_PLAIN = [t for t in old if t.lower() not in need_ev]
rel = {r["term"]: int(r["before_2019"]) for r in csv.DictReader(open(GA + "release_check.tsv"), delimiter="\t")}
kept = json.load(open(GA + "kept_terms.json"))
FLAG = [t for t in kept if rel.get(t, 0) > 0] + ["BERT", "RoBERTa"]
FLAG_LM = ["language model", "language models"]
KEPT = [t for t in kept if rel.get(t, 0) == 0]


def run(terms, field, context=None):
    Q.FIELD = field
    got = {}
    for i in range(0, len(terms), 80):
        part = terms[i:i + 80]
        if context:
            got.update(ids_ctx(part, field, context))
        else:
            got.update(Q.ids_for(part))
    return got


def ids_ctx(terms, field, context):
    """like retrieval_gap_query.ids_for, with an AND-ed context group searched in the body text"""
    import calendar
    q = ("(" + " OR ".join(f'"{t}"[{field}]' for t in terms) + ") AND (" + " OR ".join(f'"{c}"[Body - All Words]' for c in context)
         + ') AND "open access"[filter]')
    out = {}
    for y in range(2015, 2027):
        n, ids = Q.esearch(q, f"{y}/01/01", f"{y}/12/31" if y < 2026 else "2026/09/30")
        if n > 9999:
            for m in range(1, 13 if y < 2026 else 10):
                n2, ids2 = Q.esearch(q, f"{y}/{m:02d}/01", f"{y}/{m:02d}/{calendar.monthrange(y, m)[1]}")
                assert n2 <= 9999
                out.update({i: y for i in ids2})
        else:
            out.update({i: y for i in ids})
    return {f"PMC{i}": y for i, y in out.items()}


if __name__ == "__main__":
    target = set(open(LCS + "target_pmcids.txt").read().split())
    res = {"terms": dict(OLD_PLAIN=OLD_PLAIN, KEPT=KEPT, FLAG=FLAG, FLAG_LM=FLAG_LM, CONTEXT=CONTEXT,
                         excluded_need_evidence=[t for t in old if t.lower() in need_ev]), "ids": {}, "year": {}}
    jobs = [("OLD_PLAIN", OLD_PLAIN, None), ("KEPT", KEPT, None), ("FLAG_plain", FLAG + FLAG_LM, None),
            ("FLAG_context", FLAG, CONTEXT), ("FLAG_LM_context", FLAG_LM, LM_CONTEXT)]
    for name, terms, ctx in jobs:
        got = {}
        for field in ("Body - All Words", "Title/Abstract"):
            g = run(terms, field, ctx)
            got.update(g)
            print(f"{name} [{field}]: {len(g):,} articles, {len(set(g) - target):,} not in R1", flush=True)
        res["ids"][name] = sorted(got); res["year"].update(got)
    ids = {k: set(v) - target for k, v in res["ids"].items()}
    A = ids["OLD_PLAIN"] | ids["KEPT"] | ids["FLAG_plain"]
    B = ids["OLD_PLAIN"] | ids["KEPT"] | ids["FLAG_context"] | ids["FLAG_LM_context"]
    res["version_A"], res["version_B"] = sorted(A), sorted(B)
    json.dump(res, open(GA + "candidates.json", "w"))
    print(json.dumps({k: len(v) for k, v in ids.items()}), "version A", len(A), "version B", len(B), flush=True)
