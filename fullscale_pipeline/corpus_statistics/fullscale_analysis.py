"""SUPERSEDED IN PART (2026-10-02, see ../CORPUS_STATISTICS.md, "Corrections"): this pass treated
clean_corpus_v2's <PMCID>.rN.txt files as revisions and kept only the highest-numbered one. They are
region windows of the same article (P2's docstring), so for ~7,800 articles the word, sentence and
term counts below were taken from a fragment. Still used from this script: parse_header (header
author lines and Subjects field), classify_subject, and the Subjects values in fullscale_results.json.
Word/sentence counts now come from wordcount.py; focal terms from P2's focal_sentences_v2.jsonl.

Original docstring: One-pass analysis of the full-scale PMC/LLM corpus: article counts, focal-term
mention frequencies, model families, and dataset size under two filters.
Reuses pmc_preprocessing/focal_terms.py's Matcher (the project's own guarded
matching logic) rather than reimplementing term matching.
"""
import os, re, sys, json, collections
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms as ft

CLEAN = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/clean_corpus_v2/"
PURE = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus/"
OUT = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/fullscale_results.json"

matcher = None  # built once per worker process in _init_worker

# canonical display name for PLAIN hits (keys come back lowercased from Matcher.hits)
PLAIN_CANON = {w.lower(): w for w in ft.PLAIN_WORDS}
EXTRA_CANON = {  # EXTRA_PLAIN regex variants collapse to these under matching; map common lowercase forms seen
    "chatgpt": "ChatGPT", "gpt-2": "GPT-2", "gpt-3": "GPT-3", "gpt-3.5": "GPT-3.5", "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o", "gpt-4.5": "GPT-4.5", "gpt-5": "GPT-5", "gpt4": "GPT4", "gpt4o": "GPT4o",
    "gpt 4": "GPT 4", "gpt2": "GPT2", "llama-2": "Llama-2", "llama-3": "Llama-3", "llama2": "Llama2",
    "llama3": "Llama3", "llama 2": "Llama 2", "llama 3": "Llama 3",
}


def canon(term):
    if term in ft.GUARDED_NAMES or term in ft.GENERIC_NAMES:
        return term  # already canonical casing
    return PLAIN_CANON.get(term, EXTRA_CANON.get(term, term))


SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])')

# --- Subjects: / author-line parsing (adapted from diagnostics/header_authors.py) ---
STOP = re.compile(r"^(Electronic publication date|Publication date|Print publication date|Volume:|Issue:|Received|"
                   r"Copyright|Keywords:|Funding:|Article ID|Electronic Location|First page|Last page|Page count|"
                   r"Figure count|Table count|License|Data Availability|Reference count|Equation count|Word count|"
                   r"Edited by|Reviewed by)", re.I)
AFF = re.compile(r"^(\d+\s|\*|†|‡|#|email|e-mail|correspondence|contributions|these authors|equal|address|"
                  r"grid\.|ror\.|https?://ror|department|university|institute|school of|hospital|college|laborator|"
                  r"centre|center|faculty)", re.I)


def parse_header(text):
    """Returns (subjects_raw, author_lines) from a pure_text_corpus file's first ~100 lines."""
    L = text.split("\n", 120)[:120]
    subj = None
    for l in L:
        if l.startswith("Subjects:"):
            subj = l[len("Subjects:"):].strip()
            break
    if subj is None:
        return None, []
    try:
        s = next(i for i, l in enumerate(L) if l.startswith("Subjects:"))
        div = next(i for i, l in enumerate(L) if l.strip("﻿") == "\x9f" + "=" * 30 + "\x9f")
    except StopIteration:
        return subj, []
    j = s + 1
    while j < div and not L[j].strip():
        j += 1
    out = []
    for l in L[j + 1:div]:
        t = l.strip()
        if not t:
            continue
        if STOP.match(t):
            break
        if AFF.search(t):
            continue
        if re.fullmatch(r"[\d\s#*,]+", t):
            continue
        out.append(t)
    return subj, out


def classify_subject(subj):
    if subj is None:
        return "no_subjects_field"
    s = subj.lower()
    if any(k in s for k in ("letter", "correction", "erratum", "retraction", "comment", "editorial",
                             "news", "announcement")):
        return "excluded_format"
    if any(k in s for k in ("review",)):
        return "review"
    if any(k in s for k in ("article", "research", "original", "study", "report", "investigation")):
        return "article_like"
    return "other"


def read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


def _init_worker():
    global matcher
    matcher = ft.Matcher()


def process_one(job):
    pmcid, fname, in_pure = job
    body = read(CLEAN + fname)
    if body is None:
        return pmcid, None
    words = len(body.split())
    sentences = len(SENT_SPLIT.split(body.strip())) if body.strip() else 0
    anchored = bool(matcher.anchor_hits(body))
    accepted, rejected = matcher.hits(body, anchored)
    terms = collections.Counter(canon(t) for t, _, _ in accepted)

    subj, authors = None, []
    if in_pure:
        htext = read(PURE + pmcid + ".txt")
        if htext is not None:
            subj, authors = parse_header(htext[:8000])

    return pmcid, dict(words=words, sentences=sentences, anchored=anchored, terms=dict(terms),
                       n_rejected=len(rejected), subjects=subj, n_authors=len(authors))


def main():
    clean_files = sorted(os.listdir(CLEAN))
    # dedupe revisions: PMCxxxx.r2.txt supersedes PMCxxxx.txt
    by_pmcid = {}
    for f in clean_files:
        base = f.split(".")[0]
        rev = 1
        m = re.match(r"^PMC\d+\.r(\d+)\.txt$", f)
        if m:
            rev = int(m.group(1))
        prev = by_pmcid.get(base)
        if prev is None or rev > prev[1]:
            by_pmcid[base] = (f, rev)
    print(f"clean_corpus_v2: {len(clean_files)} files -> {len(by_pmcid)} distinct PMCIDs after revision dedupe", flush=True)

    pure_files = set(os.listdir(PURE))

    records = {}
    ids = list(by_pmcid)
    jobs = [(pmcid, by_pmcid[pmcid][0], (pmcid + ".txt") in pure_files) for pmcid in ids]
    done = 0
    with ProcessPoolExecutor(16, initializer=_init_worker) as ex:
        for pmcid, rec in ex.map(process_one, jobs, chunksize=64):
            if rec is not None:
                records[pmcid] = rec
            done += 1
            if done % 5000 == 0:
                print(f"  processed {done}/{len(ids)}", flush=True)

    json.dump(records, open(OUT, "w"))
    print(f"DONE. wrote {len(records)} records to {OUT}", flush=True)


if __name__ == "__main__":
    main()
