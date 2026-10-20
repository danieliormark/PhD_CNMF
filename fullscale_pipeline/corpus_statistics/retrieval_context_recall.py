# Recall of a context condition for noisy search terms (owner 2026-10-20, RL-155), measured on the current corpus: of the
# articles where P2 accepted BERT / RoBERTa / "language model(s)" (or a BERT-family name) as a focal term, the share whose
# cleaned text (P2's input) also contains a context word. Read-only.
import collections, json, os, re, sys
from multiprocessing import Pool
LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
CLEAN = LCS + "clean_corpus_v2/"
CONTEXT = [r"transformers?\b", r"language models?\b", r"natural language processing", r"\bNLP\b", r"pre-?trained", r"fine-?tun",
           r"deep learning", r"neural networks?", r"machine learning", r"embeddings?\b", r"attention mechanism", r"tokeni[sz]"]
CRX = [re.compile(c, re.I) for c in CONTEXT]
files = {}
for f in os.listdir(CLEAN):
    files.setdefault(f.split(".")[0], []).append(f)
def one(item):
    a, terms = item
    t = "\n".join(open(CLEAN + f, encoding="utf-8", errors="replace").read() for f in files.get(a, []))
    return a, terms, [bool(c.search(t)) for c in CRX]
items = []
for line in open(LCS + "focal_extractions_v2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    T = {k.lower() for k in r["terms"]}
    g = set()
    if "bert" in T: g.add("BERT")
    if "roberta" in T: g.add("RoBERTa")
    if "language model" in T: g.add("language model")
    if g: items.append((r["pmcid"], sorted(g)))
C = collections.Counter(); per = collections.Counter(); ex = []
with Pool(24) as p:
    for a, terms, hits in p.imap_unordered(one, items, chunksize=50):
        for g in terms:
            C[(g, "articles")] += 1; C[(g, "with any context word")] += any(hits)
            C[(g, "with any context word except 'language model'")] += any(h for h, c in zip(hits, CONTEXT) if "language" not in c)
        for h, c in zip(hits, CONTEXT):
            per[c] += h
        if not any(hits) and len(ex) < 10: ex.append(a)
for k in sorted(C): print(k, C[k])
print("articles without any context word (examples):", ex)
print({c: per[c] for c in CONTEXT})
